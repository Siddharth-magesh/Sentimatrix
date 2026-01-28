# Sentimatrix Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - 0.2.0

### Implemented (Phase 1-8 Complete - 2026-01-28)

#### Stage 8: Output Layer ✅
- **Exporters** (`output/exporters.py`)
  - JSONExporter with pretty print and gzip compression
  - CSVExporter with auto-column detection and nested dict flattening
  - ExcelExporter with multi-sheet support and styling (requires openpyxl)
  - ExportResult dataclass with success, path, format, size info
  - Convenience functions: export_to_json, export_to_csv, export_to_excel
  - 33 unit tests

- **Formatters** (`output/formatters.py`)
  - HTMLFormatter with responsive CSS and theme support (default, dark, colorful)
  - TextFormatter for plain text reports with separators
  - MarkdownFormatter with tables and lists
  - FormatOptions dataclass for customization
  - Convenience functions: format_as_html, format_as_text, format_as_markdown
  - 33 unit tests

- **Visualizers** (`output/visualizers.py`)
  - ChartVisualizer with matplotlib backend (Agg for non-interactive)
  - Sentiment bar chart and pie/donut charts
  - Emotion horizontal bar chart with top-k
  - Score histogram with configurable bins
  - Line charts for time series
  - Comparison charts for product comparison
  - Theme support (default, dark, colorful, minimal)
  - Save to PNG, SVG, PDF formats
  - to_bytes() for in-memory image generation
  - 29 unit tests

- **Main Class Integration**
  - export_to_json(), export_to_csv(), export_to_excel() methods
  - generate_html_report() method
  - create_sentiment_chart(), create_emotion_chart(), create_comparison_chart() methods

**Stage 8 Test Summary:** 95 new tests, 817 total tests passing

---

#### Stage 7: Pipeline Integration ✅
- **Main Sentimatrix Class** (`main.py`)
  - Unified interface for all sentiment analysis operations
  - Sentiment methods: analyze_sentiment(), analyze_sentiment_batch(), get_quick_sentiment()
  - Emotion methods: detect_emotions(), detect_emotions_batch(), detect_ekman_emotions()
  - Combined analysis: analyze(), analyze_reviews()
  - Platform scraping: scrape_amazon(), scrape_steam(), scrape_youtube(), scrape_reddit()
  - LLM integration: summarize_reviews(), generate_insights(), compare_products()
  - Pipeline integration: run_analysis_pipeline()
  - Provider management: get_provider(), set_provider()
  - 42 unit tests

**Stage 7 Test Summary:** 42 new tests, 722 total tests passing

---

#### Stage 6: Platform Scrapers ✅
- **Base Platform Scraper** (`providers/scrapers/platforms/base.py`)
  - Abstract base class for all platform scrapers
  - Common URL validation and ID extraction patterns
  - Rating normalization (convert to 0-5 scale)
  - Date parsing with multiple format support
  - Text cleaning utilities
  - Review ID generation
  - 14 unit tests

- **Amazon Scraper** (`providers/scrapers/platforms/amazon.py`)
  - ASIN validation and extraction
  - Multi-country support (us, uk, de, ca, jp, etc.)
  - Review extraction with Playwright
  - Ratings, dates, verified purchases, helpful votes
  - Product info and search functionality
  - 23 unit tests

- **Steam Scraper** (`providers/scrapers/platforms/steam.py`)
  - Steam Reviews API integration (no API key required)
  - App ID validation and extraction
  - Reviews with playtime, recommendation, timestamps
  - Game info via Store API
  - Search functionality
  - Review summary statistics
  - 17 unit tests

- **YouTube Scraper** (`providers/scrapers/platforms/youtube.py`)
  - YouTube Data API v3 integration
  - Video ID validation (11 char, alphanumeric with dashes)
  - Comment extraction with replies
  - Transcript extraction via youtube-transcript-api
  - Video info and search
  - VideoInfo and Transcript dataclasses
  - 23 unit tests

- **Reddit Scraper** (`providers/scrapers/platforms/reddit.py`)
  - Reddit JSON API integration
  - Optional OAuth authentication
  - Post and comment extraction
  - Subreddit posts and info
  - Search functionality
  - RedditPost and RedditComment dataclasses
  - 22 unit tests

**Stage 6 Test Summary:** 99 new tests, 680 total tests passing

---

#### Stage 5: Scraping Infrastructure ✅
- **Rate Limiter** (`providers/scrapers/rate_limiter.py`)
  - Token bucket algorithm for burst handling
  - Fixed window algorithm for simple rate limiting
  - Sliding window algorithm for smooth rate limiting
  - Per-domain rate limiting
  - 429 cooldown handling with automatic retry
  - Statistics tracking per domain
  - Thread-safe async implementation
  - 35 unit tests

- **Scraper Utilities** (`providers/scrapers/utils.py`)
  - ProxyManager with health tracking
  - Rotation strategies (round-robin, random, least-used, weighted, sticky)
  - Proxy configuration for HTTPX and Playwright
  - UserAgentRotator for anti-detection
  - Desktop and mobile user agent pools
  - RetryHandler with exponential backoff
  - Async retry generator with attempts control
  - Utility functions (extract_domain, normalize_url, parse_cookies)
  - 44 unit tests

- **HTTPX Scraper** (`providers/scrapers/httpx_scraper.py`)
  - Async HTTP client for static content
  - Rate limiting integration
  - Proxy rotation support
  - User agent rotation
  - Cookie management
  - Configurable timeouts and retries
  - HTML parsing with BeautifulSoup
  - Batch URL scraping
  - 16 unit tests

- **Playwright Scraper** (`providers/scrapers/playwright_scraper.py`)
  - Browser automation for JS-rendered content
  - Multiple browser support (Chromium, Firefox, WebKit)
  - Stealth mode for anti-detection
  - JavaScript execution
  - Screenshot and PDF generation
  - Page action support (click, type, scroll, wait)
  - Cookie management
  - Rate limiting integration
  - 20 unit tests

**Stage 5 Test Summary:** 115 new tests, 581 total tests passing

---

#### Stage 4: LLM Provider Integration ✅
- **OpenAI Provider** (`providers/llm/openai_provider.py`)
  - GPT-4o, GPT-4o-mini, o1 model support
  - Streaming support
  - Function calling / tool use
  - 21 unit tests

- **Anthropic Provider** (`providers/llm/anthropic_provider.py`)
  - Claude 3.5 Sonnet, Claude 3 support
  - Streaming support
  - Tool use support

- **Google Gemini Provider** (`providers/llm/gemini_provider.py`)
  - Gemini 1.5 Pro, Gemini 2.0 Flash support
  - Streaming support

- **Groq Provider** (`providers/llm/groq_provider.py`)
  - Fast inference support
  - Llama 3.3 70B, Mixtral support
  - 16 unit tests

- **Ollama Provider** (`providers/llm/ollama_provider.py`)
  - Local model support
  - Model management (pull, list, delete)

- **Provider Manager** (`providers/llm/manager.py`)
  - Provider fallback chains
  - Load balancing
  - Circuit breaker pattern
  - 22 unit tests

**Stage 4 Test Summary:** 464 tests passing

---

#### Stage 3: Sentiment Analysis Core ✅
- **HuggingFace Model Provider** (`providers/models/huggingface.py`)
  - Automatic device detection (CPU/CUDA/MPS)
  - Model caching for efficiency
  - Batch processing support
  - SentimentModelProvider and EmotionModelProvider specializations
  - 21 unit tests

- **Sentiment Analysis** (`analysis/sentiment.py`)
  - SentimentAnalyzer class with 3-class and 5-class modes
  - SentimentResult and BatchSentimentResult data classes
  - Label normalization for various model outputs
  - Polarity calculation (-1 to 1 scale)
  - Async and sync convenience functions
  - 40 unit tests, 92% coverage

- **Emotion Detection** (`analysis/emotion.py`)
  - EmotionDetector class with multi-label, top-k, and single-label modes
  - GoEmotions (28 emotions) support
  - Ekman's 6 basic emotions mapping
  - Plutchik's 8 emotions support
  - Emotion valence classification (positive/negative/neutral)
  - Batch processing with aggregate statistics
  - 54 unit tests, 91% coverage

**Test Summary:** 405 tests passing, 91% overall coverage

---

#### Core Infrastructure ✅
- **Configuration System** (`core/config.py`) - Pydantic v2 based
  - YAML/JSON file loading
  - Environment variable support with `SENTIMATRIX_` prefix
  - Runtime overrides
  - Full validation
  - 50 unit tests, 98% coverage

- **Exception Hierarchy** (`core/exceptions.py`)
  - 50+ error types with error codes
  - Provider-specific errors (OpenAI, Anthropic, Groq, etc.)
  - Serializable exceptions
  - 44 unit tests, 98% coverage

- **Logging System** (`core/logger.py`)
  - Structured JSON/text formatters
  - Process/thread ID tracking
  - Context propagation
  - Timing decorators and context managers
  - Performance logging helpers
  - Rich colorized console output
  - Rotating file handlers
  - 52 unit tests, 87% coverage

- **Cache Layer** (`core/cache.py`)
  - Memory cache with LRU eviction
  - TTL support
  - Compression support
  - Cache statistics
  - @cached decorator
  - get_or_set pattern
  - 43 unit tests, 84% coverage

- **Pipeline Orchestration** (`core/pipeline.py`)
  - Step chaining with data flow
  - Parallel step execution
  - Conditional step execution
  - Progress callbacks
  - Error handling with retry support
  - Step timeout support
  - Lifecycle hooks (on_start, on_complete, on_error)
  - 64 unit tests, 89% coverage

- **Provider Interfaces** (`providers/base.py`)
  - BaseLLMProvider abstract class
  - BaseScraperProvider abstract class
  - BaseModelProvider abstract class
  - Provider registry with discovery
  - Data models (LLMResponse, Review, PredictionResult)
  - 37 unit tests, 88% coverage

**Core Test Summary:** 290 tests passing (core only), 91% overall coverage

---

### Planned (Not Yet Implemented)

#### Core
- CLI interface

#### LLM Providers (Additional)
- Mistral provider
- Cohere provider
- Together AI provider
- Fireworks AI provider
- DeepSeek provider
- Cerebras provider
- SambaNova provider
- vLLM provider (local)

#### Scrapers (Commercial)
- Selenium scraper (legacy)
- ScraperAPI integration
- Bright Data integration
- Oxylabs integration
- Apify integration
- Zyte integration
- AI-powered scrapers (Firecrawl, Crawl4AI)

#### Platform Scrapers (P1+)
- IMDB reviews
- Yelp reviews
- Trustpilot reviews
- Google Reviews
- Twitter/X
- Metacritic
- Rotten Tomatoes
- LetterBoxD
- App Store
- Play Store
- TikTok
- Tripadvisor
- Glassdoor
- LinkedIn

#### Analysis
- Batch sentiment processing
- Emotion detection with GoEmotions
- Aspect-based sentiment analysis
- Comparative analysis
- Temporal sentiment tracking
- Multi-lingual support

#### Multi-Modal
- Audio sentiment (via Whisper)
- Image sentiment (via LLaVA/GPT-4V)
- Video analysis

#### Output
- Webhook support (HTTP callbacks)

### Changed
- Complete architecture rewrite
- New configuration format (YAML-based)
- All I/O operations now async
- Improved error handling
- Better memory management

### Deprecated
- Old `SentConfig` class (use `Sentimatrix` instead)
- Synchronous API methods

### Removed
- Python 3.9 support (now requires 3.10+)

### Fixed
- Memory leaks in long-running processes
- Rate limiting issues with APIs
- Unicode handling in reviews

### Security
- API keys no longer logged
- Input sanitization improved
- Dependency updates

---

## [0.1.7] - Previous Release

### Features
- Basic sentiment analysis
- Emotion detection
- Web scraping (7 platforms)
- Groq, Gemini, Ollama, OpenAI (partial) support
- CSV export
- Basic visualizations
- Audio to text
- Image to text

### Limitations
- Synchronous API
- Limited error handling
- Basic configuration
- Minimal testing

---

## Version History

| Version | Date | Status |
|---------|------|--------|
| 0.2.0 | TBD | In Development |
| 0.1.7 | 2024 | Previous Release |
| 0.1.0 | 2024 | Initial Release |

---

## Migration Guide (0.1.x to 0.2.0)

### Configuration Changes

**Before (0.1.x):**
```python
from Sentimatrix.sentiment_generation import SentConfig

sent = SentConfig(
    Use_Local_Sentiment_LLM=True,
    Use_Groq_API=True,
    Groq_API="gsk_...",
    Groq_LLM="llama3-8b-8192"
)
```

**After (0.2.0):**
```python
from sentimatrix import Sentimatrix

sm = Sentimatrix(config={
    "llm": {
        "provider": "groq",
        "api_key": "gsk_...",
        "model": "llama-3.3-70b-versatile"
    }
})
```

### API Changes

**Before:**
```python
result = sent.get_Quick_sentiment("Great product!")
```

**After:**
```python
result = await sm.analyze_sentiment("Great product!")
# or sync wrapper
result = sm.analyze_sentiment_sync("Great product!")
```

### Import Changes

| 0.1.x | 0.2.0 |
|-------|-------|
| `Sentimatrix.sentiment_generation` | `sentimatrix` |
| `SentConfig` | `Sentimatrix` |
| `get_Quick_sentiment()` | `analyze_sentiment()` |
| `get_emotion_from_website_each_feedback()` | `detect_emotions()` |
