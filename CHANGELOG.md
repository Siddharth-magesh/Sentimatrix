# Changelog

All notable changes to Sentimatrix will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

#### Stage 12: Documentation & CI/CD (2026-01-28)

- **API Reference Documentation** (`docs/api/README.md`)
  - Comprehensive API documentation for all modules
  - Core module (Pipeline, Config, Exceptions)
  - Analysis module (Sentiment, Emotion, Aggregation)
  - Providers module (LLM, Scrapers)
  - Input/Output modules
  - Utils module (Cache, Logging, Text)
  - Type definitions and error handling

- **Usage Guides** (`docs/guides/`)
  - `quickstart.md` - Getting started guide
  - `examples.md` - Comprehensive code examples
    - Basic analysis
    - Batch processing
    - Pipeline workflows
    - Web scraping
    - Multi-modal analysis
    - Advanced configurations
    - Real-world use cases
  - `troubleshooting.md` - Common issues and solutions
    - Installation issues
    - Import errors
    - Provider errors
    - Scraper issues
    - Pipeline errors
    - Performance issues
    - Cache issues

- **CI/CD Workflows** (`.github/workflows/`)
  - `ci.yml` - Main CI/CD pipeline
    - Lint & format checks (ruff, black, isort, mypy)
    - Unit tests with matrix (Python 3.9-3.12)
    - Integration tests with Redis service
    - E2E tests
    - Security scan (bandit, safety)
    - Package build and verification
    - Documentation build
    - PyPI release (on tag)
    - Codecov integration
  - `release.yml` - Release workflow
    - Version validation
    - Full test suite
    - Version bump automation
    - GitHub release creation
    - PyPI/TestPyPI publishing

---

#### Stage 11: Integration & E2E Tests (2026-01-28)

- **Integration Tests** (`tests/integration/`)
  - `test_pipeline_integration.py` - Pipeline workflow tests
    - Function step chaining
    - Context propagation
    - Error handling
    - Parallel step execution
  - `test_provider_integration.py` - Provider integration tests
    - Sentiment analyzer with mocked providers
    - Emotion detector with mocked providers
    - Combined analysis tests
    - Error recovery tests
    - Concurrent access tests
  - `test_scraper_integration.py` - Scraper integration tests
    - Rate limiter integration
    - Data flow processing
    - Scraper + analysis pipeline
    - Concurrent fetching

- **E2E Tests** (`tests/e2e/`)
  - `test_full_workflow.py` - Complete workflow tests
    - Text analysis workflow
    - Review analysis workflow
    - Export workflows (JSON, CSV)
    - Pipeline workflow
    - Error handling workflow
    - Caching workflow
    - Multi-modal workflow (audio, image, combined)

---

#### Stage 10: Multi-Modal Processing (2026-01-28)

- **Input Handlers** (`sentimatrix/input/handlers.py`)
  - `AudioHandler` - Audio transcription with multiple engines
    - Local Whisper (openai-whisper)
    - OpenAI Whisper API
    - Groq Whisper (fast cloud inference)
  - `ImageHandler` - Image captioning with multiple models
    - GPT-4V (OpenAI Vision)
    - Claude Vision (Anthropic)
    - Gemini Vision (Google)
    - BLIP (local HuggingFace)
  - `VideoHandler` - Video frame extraction
    - OpenCV-based frame extraction
    - Uniform and keyframe extraction modes
    - Audio extraction support
  - Data classes: `TranscriptionResult`, `CaptionResult`, `VideoFrameResult`
  - Supported formats: Audio (wav, mp3, flac, etc.), Image (png, jpg, webp, etc.), Video (mp4, avi, mov, etc.)
  - 33 unit tests

- **Multi-Modal Analysis** (`sentimatrix/analysis/multimodal.py`)
  - `MultiModalAnalyzer` - Combined audio/image/video sentiment analysis
  - Fusion strategies: `late` (majority vote), `weighted` (configurable weights), `dominant` (highest confidence)
  - `AudioAnalysisResult` - Transcription with sentiment/emotion
  - `ImageAnalysisResult` - Caption with sentiment/emotion
  - `VideoAnalysisResult` - Frame analysis with timeline
  - `MultiModalResult` - Combined multi-modal analysis
  - Async context manager support
  - 27 unit tests

- **Main Sentimatrix Class Integration**
  - `analyze_audio()` - Audio sentiment analysis
  - `analyze_image()` - Image sentiment analysis
  - `analyze_video()` - Video sentiment analysis with frame/audio analysis
  - `transcribe_audio()` - Audio transcription without sentiment
  - `caption_image()` - Image captioning without sentiment
  - `analyze_multimodal()` - Combined text/audio/image analysis

**Test Summary:** 938 tests passing (60 multi-modal + 38 integration/E2E tests)

---

#### Stage 9: Redis Cache (2026-01-28)

- **Redis Cache Backend** (`sentimatrix/core/cache.py`)
  - `RedisCache` class for distributed caching
  - Connection pooling with configurable max connections
  - Async operations: `get`, `set`, `delete`, `exists`, `clear`
  - Batch operations: `get_many`, `set_many`, `delete_many`
  - TTL operations: `ttl`, `expire`
  - Increment operation for counters
  - Health check support
  - Compression support with zlib
  - Serialization with pickle (JSON fallback)
  - Cache statistics tracking
  - 20 unit tests

**Test Summary:** 840 tests passing (20 new Redis cache tests)

---

#### Stage 8: Output Layer (2026-01-28)

- **Export Functions** (`sentimatrix/output/exporters.py`)
  - `JSONExporter` - JSON export with compression and pretty-print
  - `CSVExporter` - CSV export with auto-columns and flattening
  - `ExcelExporter` - Excel export with multi-sheet and styling
  - Convenience functions: `export_to_json`, `export_to_csv`, `export_to_excel`

- **Formatters** (`sentimatrix/output/formatters.py`)
  - `HTMLFormatter` - HTML reports with themes (light, dark, colorful)
  - `TextFormatter` - Plain text output
  - `MarkdownFormatter` - Markdown tables and lists

- **Visualizations** (`sentimatrix/output/visualizers.py`)
  - `ChartVisualizer` - Matplotlib-based chart generation
  - Bar charts (sentiment, generic)
  - Pie/donut charts
  - Histograms (score distribution)
  - Horizontal bar charts (emotions)
  - Line charts (time series)
  - Comparison charts (multi-product)
  - Save support: PNG, SVG, PDF
  - 95 unit tests

**Test Summary:** 817 tests passing (95 new output tests)

---

#### Stage 7: Pipeline Integration (2026-01-28)

- **Main Sentimatrix Class** (`sentimatrix/main.py`)
  - Unified interface for all Sentimatrix functionality
  - Constructor with flexible configuration (dict, SentimatrixConfig, overrides)
  - Async context manager support for automatic resource cleanup
  - Lazy initialization of scrapers for efficient resource usage

- **Sentiment Analysis Methods**
  - `analyze_sentiment()` - Single text sentiment analysis
  - `analyze_sentiment_batch()` - Batch sentiment analysis
  - `get_quick_sentiment()` - Fast label+score return

- **Emotion Detection Methods**
  - `detect_emotions()` - Multi-label emotion detection
  - `detect_emotions_batch()` - Batch emotion detection
  - `detect_ekman_emotions()` - Ekman's 6 basic emotions mapping

- **Combined Analysis Methods**
  - `analyze()` - Full analysis (sentiment + emotions) for single text
  - `analyze_reviews()` - Batch analysis with aggregate statistics

- **Platform Scraping Methods**
  - `scrape_amazon()` - Amazon product reviews
  - `scrape_steam()` - Steam game reviews
  - `scrape_youtube()` - YouTube video comments
  - `scrape_reddit()` - Reddit post comments

- **LLM Integration Methods**
  - `summarize_reviews()` - Generate review summaries with style options
  - `generate_insights()` - Extract pros, cons, recommendations, themes
  - `compare_products()` - Compare two products based on reviews

- **Pipeline Integration**
  - `run_analysis_pipeline()` - Full end-to-end analysis pipeline
  - Platform auto-detection from URLs and identifiers
  - Identifier extraction from various URL formats

- **Result Dataclasses**
  - `AnalysisResult` - Combined sentiment + emotion for single text
  - `ReviewAnalysisResult` - Aggregated review analysis with ratios
  - `InsightsResult` - LLM-generated insights structure
  - `ComparisonResult` - Product comparison results

**Test Summary:** 722 tests passing (42 new Sentimatrix class tests)

---

#### Stage 6: Platform Scrapers (2026-01-28)

- **Base Platform Scraper** (`sentimatrix/providers/scrapers/platforms/base.py`)
  - Abstract base class for all platform scrapers
  - Common URL validation and ID extraction patterns
  - Rating normalization (convert to 0-5 scale)
  - Date parsing with multiple format support
  - Text cleaning utilities
  - Review ID generation
  - 14 unit tests

- **Amazon Scraper** (`sentimatrix/providers/scrapers/platforms/amazon.py`)
  - ASIN validation (10 alphanumeric characters)
  - Multi-country support (us, uk, de, ca, jp, fr, it, es, in, au)
  - Review extraction with Playwright browser automation
  - Ratings, dates, verified purchases, helpful votes
  - Product info and search functionality
  - 23 unit tests

- **Steam Scraper** (`sentimatrix/providers/scrapers/platforms/steam.py`)
  - Steam Reviews API integration (no API key required)
  - App ID validation and URL extraction
  - Reviews with playtime, recommendation, timestamps
  - Game info via Steam Store API
  - Search functionality
  - Review summary statistics
  - 17 unit tests

- **YouTube Scraper** (`sentimatrix/providers/scrapers/platforms/youtube.py`)
  - YouTube Data API v3 integration
  - Video ID validation (11 chars, alphanumeric with dashes)
  - Comment extraction with nested replies
  - Transcript extraction via youtube-transcript-api
  - Video info and search functionality
  - VideoInfo and Transcript dataclasses
  - 23 unit tests

- **Reddit Scraper** (`sentimatrix/providers/scrapers/platforms/reddit.py`)
  - Reddit JSON API integration (no OAuth required)
  - Optional OAuth authentication for higher rate limits
  - Post and comment extraction with nested replies
  - Subreddit posts and info
  - Search functionality
  - RedditPost and RedditComment dataclasses
  - 22 unit tests

**Test Summary:** 680 tests passing (99 new platform scraper tests)

---

#### Stage 5: Scraping Infrastructure (2026-01-28)

- **Rate Limiter** (`sentimatrix/providers/scrapers/rate_limiter.py`)
  - Token bucket algorithm for burst handling
  - Fixed window algorithm for simple rate limiting
  - Sliding window algorithm for smooth limiting
  - Per-domain rate limiting
  - 429 cooldown handling with automatic retry
  - Statistics tracking per domain
  - 35 unit tests

- **Scraper Utilities** (`sentimatrix/providers/scrapers/utils.py`)
  - ProxyManager with health tracking and rotation
  - Rotation strategies: round-robin, random, least-used, weighted, sticky
  - UserAgentRotator for anti-detection
  - RetryHandler with exponential backoff
  - Utility functions: extract_domain, normalize_url, parse_cookies
  - 44 unit tests

- **HTTPX Scraper** (`sentimatrix/providers/scrapers/httpx_scraper.py`)
  - Async HTTP client for static content
  - Rate limiting integration
  - Proxy and user agent rotation
  - Cookie management
  - HTML parsing with BeautifulSoup
  - Batch URL scraping
  - 16 unit tests

- **Playwright Scraper** (`sentimatrix/providers/scrapers/playwright_scraper.py`)
  - Browser automation for JS-rendered content
  - Multiple browser support (Chromium, Firefox, WebKit)
  - Stealth mode for anti-detection
  - Screenshot and PDF generation
  - Page actions (click, type, scroll, wait)
  - 20 unit tests

**Test Summary:** 581 tests passing (115 new scraping infrastructure tests)

---

#### Stage 4: LLM Providers (2026-01-28)

- **OpenAI Provider** (`sentimatrix/providers/llm/openai_provider.py`)
  - Full Chat Completions API support (GPT-4o, GPT-4o-mini, o1)
  - Streaming responses
  - Function calling (tool use)
  - Text embeddings (text-embedding-3-small)
  - Token counting with tiktoken
  - 128K context window support
  - Vision support for multimodal models
  - 21 unit tests

- **Groq Provider** (`sentimatrix/providers/llm/groq_provider.py`)
  - Ultra-fast inference (~750 tokens/second)
  - LLaMA 3.3/3.2/3.1, Mixtral, Gemma models
  - Streaming responses
  - Function calling support
  - Whisper audio transcription
  - Free tier support (30 req/min)
  - 16 unit tests

- **Anthropic Provider** (`sentimatrix/providers/llm/anthropic_provider.py`)
  - Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
  - 200K context window
  - Streaming responses
  - Tool use (function calling)
  - Vision support (image inputs)
  - OpenAI-compatible tool format conversion

- **Ollama Provider** (`sentimatrix/providers/llm/ollama_provider.py`)
  - Local model inference (no API key required)
  - HTTP API client for localhost:11434
  - Streaming responses
  - Chat and generate modes
  - Embeddings support
  - Model pull/list management
  - Support for LLaMA, Mistral, Mixtral, Phi, Gemma, etc.

- **Google Gemini Provider** (`sentimatrix/providers/llm/gemini_provider.py`)
  - Gemini 2.0 Flash, 1.5 Pro/Flash models
  - Up to 2M token context (Gemini 1.5 Pro)
  - Streaming responses
  - Function calling
  - Vision support (multimodal)
  - Text embeddings
  - Free tier available

- **LLM Provider Manager** (`sentimatrix/providers/llm/manager.py`)
  - Multi-provider orchestration
  - Automatic fallback on errors
  - Health tracking per provider
  - Multiple fallback strategies:
    - Sequential (priority-based)
    - Round-robin (load balancing)
    - Fastest (response time)
    - Cheapest (priority-based)
  - Rate limit handling with backoff
  - 22 unit tests

- **Unit Tests** (`tests/unit/providers/llm/`)
  - OpenAI provider tests (21 test cases)
  - Groq provider tests (16 test cases)
  - Provider manager tests (22 test cases)

**Test Summary:** 464 tests passing (100% pass rate)

#### Stage 3: Sentiment Analysis Core (2026-01-28)

- **HuggingFace Model Provider** (`sentimatrix/providers/models/huggingface.py`)
  - `HuggingFaceModelProvider` - Base provider for HuggingFace Transformers models
  - `SentimentModelProvider` - Specialized for sentiment analysis
  - `EmotionModelProvider` - Specialized for emotion detection with top-k and multi-label modes
  - Automatic device detection (CPU/CUDA/MPS)
  - Model caching with global cache
  - Batch processing support

- **Sentiment Analysis** (`sentimatrix/analysis/sentiment.py`)
  - `SentimentAnalyzer` - Main analyzer class
  - `SentimentResult` - Result dataclass with polarity calculation
  - `BatchSentimentResult` - Batch result with aggregate statistics
  - 3-class (positive/neutral/negative) and 5-class classification modes
  - Label normalization for various model outputs
  - Async and sync convenience functions

- **Emotion Detection** (`sentimatrix/analysis/emotion.py`)
  - `EmotionDetector` - Main detector class
  - `EmotionResult` - Result with Ekman mapping and valence
  - `BatchEmotionResult` - Batch result with emotion distribution
  - GoEmotions (28 classes) support
  - Ekman's 6 basic emotions mapping
  - Detection modes: single-label, multi-label, top-k
  - Emotion valence classification

- **Unit Tests**
  - HuggingFace provider tests (21 test cases)
  - Sentiment analysis tests (40 test cases)
  - Emotion detection tests (54 test cases)

#### Stage 1: Foundation (2024-01-28)

- **Project Structure**
  - Created `pyproject.toml` with all dependencies and tool configurations
  - Setup modular directory structure following Python best practices
  - Configured development tools: ruff, black, mypy, pytest
  - Added comprehensive `.gitignore`

- **Configuration System** (`sentimatrix/core/config.py`)
  - Implemented `SentimatrixConfig` main configuration class using Pydantic v2
  - Added sub-configurations: `LLMConfig`, `ScraperConfig`, `ModelConfig`, `CacheConfig`, `LogConfig`
  - YAML/JSON file loading with `from_file()` method
  - Environment variable support with `SENTIMATRIX_` prefix
  - Runtime configuration overrides with `with_overrides()`
  - Configuration validation and serialization

- **Exception Hierarchy** (`sentimatrix/core/exceptions.py`)
  - Implemented `SentimatrixError` base exception with error codes
  - Configuration errors: `ConfigurationError`, `ConfigNotFoundError`, `ConfigValidationError`
  - Validation errors: `ValidationError`, `InvalidInputError`, `MissingRequiredFieldError`
  - Provider errors: `ProviderError`, `LLMProviderError`, `ScraperError`, `ModelError`
  - Provider-specific errors: `OpenAIError`, `AnthropicError`, `GroqError`, `PlaywrightError`
  - Rate limiting: `RateLimitError`, `QuotaExceededError`
  - Timeout errors: `TimeoutError`, `ConnectionTimeoutError`, `ReadTimeoutError`
  - Cache errors: `CacheError`, `CacheConnectionError`, `CacheReadError`, `CacheWriteError`
  - Pipeline errors: `PipelineError`, `PipelineStepError`, `PipelineStateError`

- **Logging System** (`sentimatrix/core/logger.py`)
  - Implemented `StructuredLogger` with JSON and text formatters
  - `LogContext` context manager for request/correlation ID propagation
  - `LogManager` singleton for centralized log configuration
  - Console and rotating file handlers
  - Colorized output support using Rich library
  - `BoundLogger` for contextual logging with persistent bindings

- **Cache Layer** (`sentimatrix/core/cache.py`)
  - Implemented `MemoryCache` with LRU eviction
  - TTL (time-to-live) support with automatic expiration
  - `CacheManager` high-level interface with namespace support
  - Cache statistics tracking (hits, misses, evictions)
  - Optional compression support
  - `@cached` decorator for function memoization
  - `get_or_set()` pattern for compute-on-miss

- **Provider Interfaces** (`sentimatrix/providers/base.py`)
  - `BaseLLMProvider` abstract class for LLM providers
  - `BaseScraperProvider` abstract class for web scrapers
  - `BaseModelProvider` abstract class for ML models
  - `ProviderRegistry` for dynamic provider discovery
  - Data models: `LLMResponse`, `ScrapedContent`, `Review`, `PredictionResult`
  - `ProviderCapabilities` for feature detection
  - `TokenUsage` for tracking API usage

- **Pipeline Orchestration** (`sentimatrix/core/pipeline.py`)
  - Implemented `Pipeline` class for step chaining and orchestration
  - Added `PipelineStep` abstract base class for custom steps
  - Added `FunctionStep` for wrapping functions as steps
  - Added `ParallelSteps` for parallel execution
  - Added `ConditionalStep` for conditional execution
  - Implemented `PipelineContext` for shared state
  - Added progress callbacks for monitoring
  - Implemented error handling with retry support (exponential backoff)
  - Added step timeout support
  - Added lifecycle hooks (on_start, on_complete, on_error)

- **Unit Tests**
  - Comprehensive tests for configuration system (50+ test cases)
  - Exception hierarchy tests with serialization
  - Logger tests including context propagation
  - Cache tests with TTL and eviction scenarios
  - Provider interface and registry tests
  - Pipeline orchestration tests (64 test cases)
  - Test fixtures and sample data
  - Test runner script with coverage support

### Technical Details

- Python 3.10+ required
- Async-first design with `asyncio`
- Type hints throughout codebase
- Pydantic v2 for data validation
- Frozen/immutable configuration models

---

## [0.1.0] - Previous Version

- Initial release (Sentimatrix V1)
