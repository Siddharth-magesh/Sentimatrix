# Changelog

All notable changes to Sentimatrix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

#### Stage 14: Commercial Scraping APIs (2026-01-29)

- **Commercial API Clients** (`sentimatrix/providers/scrapers/commercial/`)
  - `BaseCommercialClient` - Abstract base class for all commercial scrapers
  - `ScraperAPIClient` - ScraperAPI integration (40M+ proxies, JS rendering, CAPTCHA)
  - `ApifyClient` - Apify integration (2000+ actors, dataset management)
  - `BrightDataClient` - Bright Data integration (72M+ proxies, platform scrapers)
  - `OxylabsClient` - Oxylabs integration (100M+ proxies, e-commerce focus)
  - `ZyteClient` - Zyte integration (AI extraction, browser rendering)
  - `ScrapingBeeClient` - ScrapingBee integration (screenshots, AI extraction)
  - `ScrapingAntClient` - ScrapingAnt integration (budget option, markdown output)
  - 40 unit tests

- **Additional Platform Scrapers** (`sentimatrix/providers/scrapers/platforms/`)
  - `IMDBScraper` - IMDB movie/TV reviews
  - `YelpScraper` - Yelp business reviews
  - `TrustpilotScraper` - Trustpilot company reviews
  - `GoogleReviewsScraper` - Google Maps/Places reviews
  - 26 unit tests

- **Bug Fixes**
  - Fixed `ScraperConnectionError` missing `provider` argument in HTTPXScraper
  - Fixed `scrape_reviews` abstract method in BaseCommercialClient

**Stage 14 Test Summary:** 66 new tests, 282 total tests passing

---

#### Stage 13: CLI Interface (2026-01-29)

- **Command Line Interface** (`sentimatrix/cli.py`)
  - Full-featured CLI for Sentimatrix operations
  - Commands:
    - `analyze` - Analyze sentiment of a single text
    - `analyze-file` - Batch analyze texts from file (txt, csv, json)
    - `scrape` - Scrape reviews from platforms (Amazon, Steam, YouTube, Reddit)
    - `batch` - Process CSV files with sentiment analysis
    - `info` - Display system information and dependencies
  - Features:
    - Rich terminal output with tables and progress bars (optional)
    - JSON and CSV output formats
    - Emotion detection support
    - Platform-specific scraping with rate limiting
  - 28 unit tests

- **CI/CD Workflows** (`.github/workflows/`)
  - `ci.yml` - Comprehensive CI pipeline
    - Lint & format (ruff, black, isort, mypy)
    - Unit tests (Python 3.10-3.12 matrix)
    - Integration tests with Redis
    - E2E tests
    - Security scan (bandit, safety)
    - Package build
    - Codecov integration
    - PyPI release automation
  - `release.yml` - Manual release workflow
    - Version validation
    - Full test suite
    - Version bump automation
    - GitHub release creation
    - PyPI/TestPyPI publishing

---

#### Stage 12: Documentation & Additional LLM Providers (2026-01-28)

- **Additional LLM Providers** (`sentimatrix/providers/llm/`)
  - `MistralProvider` - Mistral AI (7B, 8x7B, Large models)
  - `CerebrasProvider` - Cerebras ultra-fast inference
  - `FireworksProvider` - Fireworks AI serverless
  - `TogetherProvider` - Together AI (200+ models)
  - `OpenRouterProvider` - OpenRouter aggregator
  - `CohereProvider` - Cohere (Command, Embed)
  - `LMStudioProvider` - LM Studio local server
  - `vLLMProvider` - vLLM high-performance server
  - `DeepSeekProvider` - DeepSeek (coder models)
  - `LlamaCppProvider` - llama.cpp GGUF models
  - `TextGenProvider` - text-generation-webui
  - `ExLlamaV2Provider` - ExLlamaV2 quantized
  - `AzureOpenAIProvider` - Azure OpenAI Service
  - `BedrockProvider` - AWS Bedrock
  - 112 unit tests for all providers

- **API Reference Documentation** (`docs/api/README.md`)
  - Comprehensive API documentation for all modules

- **Usage Guides** (`docs/guides/`)
  - `quickstart.md` - Getting started guide
  - `examples.md` - Comprehensive code examples
  - `troubleshooting.md` - Common issues and solutions

---

#### Stage 11: Integration & E2E Tests (2026-01-28)

- **Integration Tests** (`tests/integration/`)
  - Pipeline workflow tests
  - Provider integration tests
  - Scraper integration tests

- **E2E Tests** (`tests/e2e/`)
  - Complete workflow tests
  - Multi-modal workflow tests

---

#### Stage 10: Multi-Modal Processing (2026-01-28)

- **Input Handlers** (`sentimatrix/input/handlers.py`)
  - `AudioHandler` - Audio transcription (Whisper, OpenAI, Groq)
  - `ImageHandler` - Image captioning (GPT-4V, Claude Vision, Gemini, BLIP)
  - `VideoHandler` - Video frame extraction
  - 33 unit tests

- **Multi-Modal Analysis** (`sentimatrix/analysis/multimodal.py`)
  - `MultiModalAnalyzer` - Combined audio/image/video analysis
  - Fusion strategies: late, weighted, dominant
  - 27 unit tests

---

#### Stage 9: Redis Cache (2026-01-28)

- **Redis Cache Backend** (`sentimatrix/core/cache.py`)
  - `RedisCache` class for distributed caching
  - Connection pooling, TTL, compression
  - 20 unit tests

---

#### Stage 8: Output Layer (2026-01-28)

- **Exporters** (`sentimatrix/output/exporters.py`)
  - JSON, CSV, Excel export with compression

- **Formatters** (`sentimatrix/output/formatters.py`)
  - HTML, Text, Markdown formatters

- **Visualizers** (`sentimatrix/output/visualizers.py`)
  - Bar, pie, histogram, line, comparison charts
  - 95 unit tests

---

#### Stage 7: Pipeline Integration (2026-01-28)

- **Main Sentimatrix Class** (`sentimatrix/main.py`)
  - Unified interface for all functionality
  - Sentiment, emotion, scraping, LLM integration
  - 42 unit tests

---

#### Stage 6: Platform Scrapers (2026-01-28)

- **Platform Scrapers** (`sentimatrix/providers/scrapers/platforms/`)
  - `AmazonScraper` - Amazon reviews (multi-country)
  - `SteamScraper` - Steam game reviews (JSON API)
  - `YouTubeScraper` - YouTube comments (Data API v3)
  - `RedditScraper` - Reddit posts/comments (JSON API)
  - 99 unit tests

---

#### Stage 5: Scraping Infrastructure (2026-01-28)

- **Rate Limiter** (`sentimatrix/providers/scrapers/rate_limiter.py`)
  - Token bucket, fixed window, sliding window algorithms
  - 35 unit tests

- **Scraper Utilities** (`sentimatrix/providers/scrapers/utils.py`)
  - ProxyManager, UserAgentRotator, RetryHandler
  - 44 unit tests

- **Core Scrapers**
  - `HTTPXScraper` - Async HTTP client
  - `PlaywrightScraper` - Browser automation
  - 36 unit tests

---

#### Stage 4: LLM Providers (2026-01-28)

- **Core Providers** (`sentimatrix/providers/llm/`)
  - `OpenAIProvider` - GPT-4o, o1 models
  - `GroqProvider` - Ultra-fast LLaMA, Mixtral
  - `AnthropicProvider` - Claude 3.5 Sonnet
  - `OllamaProvider` - Local models
  - `GeminiProvider` - Google Gemini
  - `LLMProviderManager` - Multi-provider orchestration
  - 59 unit tests

---

#### Stage 3: Sentiment Analysis Core (2026-01-28)

- **HuggingFace Provider** (`sentimatrix/providers/models/huggingface.py`)
  - SentimentModelProvider, EmotionModelProvider
  - 21 unit tests

- **Sentiment Analysis** (`sentimatrix/analysis/sentiment.py`)
  - 3-class and 5-class classification
  - 40 unit tests

- **Emotion Detection** (`sentimatrix/analysis/emotion.py`)
  - GoEmotions (28 emotions), Ekman mapping
  - 54 unit tests

---

#### Stage 1-2: Foundation (2026-01-28)

- **Configuration System** (`sentimatrix/core/config.py`)
  - Pydantic v2 based, YAML/JSON loading
  - 50 unit tests

- **Exception Hierarchy** (`sentimatrix/core/exceptions.py`)
  - 50+ error types with error codes
  - 44 unit tests

- **Logging System** (`sentimatrix/core/logger.py`)
  - Structured JSON/text logging
  - 52 unit tests

- **Cache Layer** (`sentimatrix/core/cache.py`)
  - Memory cache with LRU eviction
  - 43 unit tests

- **Pipeline Orchestration** (`sentimatrix/core/pipeline.py`)
  - Step chaining, parallel execution
  - 64 unit tests

- **Provider Interfaces** (`sentimatrix/providers/base.py`)
  - Base classes and data models
  - 37 unit tests

---

## [0.1.0] - Previous Version

- Initial release (Sentimatrix V1)

---

## Version History

| Version | Date | Status | Tests |
|---------|------|--------|-------|
| 0.2.0 | 2026-01-29 | In Development | 282 |
| 0.1.7 | 2024 | Previous Release | - |
| 0.1.0 | 2024 | Initial Release | - |
