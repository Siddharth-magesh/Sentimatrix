# Sentimatrix V2 - Architecture Overview

## Version: 0.2.0

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            SENTIMATRIX V2                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   INPUT     │  │  SCRAPING   │  │  ANALYSIS   │  │   OUTPUT    │        │
│  │   LAYER     │  │   LAYER     │  │   LAYER     │  │   LAYER     │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         ▼                ▼                ▼                ▼                │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                     CORE ENGINE                                  │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │       │
│  │  │ Config   │ │ Pipeline │ │ Cache    │ │ Logger   │           │       │
│  │  │ Manager  │ │ Manager  │ │ Manager  │ │ Manager  │           │       │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐       │
│  │                     PROVIDER LAYER                               │       │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │       │
│  │  │ LLM      │ │ Scraper  │ │ Model    │ │ Storage  │           │       │
│  │  │ Providers│ │ Providers│ │ Providers│ │ Providers│           │       │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │       │
│  └─────────────────────────────────────────────────────────────────┘       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Principles

### 1. Modularity
- Each component operates independently
- Providers are interchangeable
- Easy to add new scrapers, models, or LLM providers

### 2. Configuration-Driven
- YAML/JSON configuration files
- Environment variable support
- Runtime configuration override

### 3. Provider Abstraction
- Unified interface for all LLM providers
- Unified interface for all scraper providers
- Unified interface for all model providers

### 4. Async-First Design
- Asynchronous scraping operations
- Concurrent API calls
- Non-blocking I/O operations

### 5. Extensibility
- Plugin architecture for custom providers
- Hook system for pipeline customization
- Event-driven architecture

## Directory Structure

```
Sentimatrix/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── config.py           # Configuration management
│   ├── pipeline.py         # Pipeline orchestration
│   ├── cache.py            # Caching layer
│   ├── logger.py           # Logging utilities
│   └── exceptions.py       # Custom exceptions
├── providers/
│   ├── __init__.py
│   ├── base.py             # Base provider classes
│   ├── llm/                # LLM providers
│   ├── scrapers/           # Scraper providers
│   ├── models/             # ML model providers
│   └── storage/            # Storage providers
├── analysis/
│   ├── __init__.py
│   ├── sentiment.py        # Sentiment analysis
│   ├── emotion.py          # Emotion detection
│   ├── aspect.py           # Aspect-based analysis
│   └── multimodal.py       # Multi-modal analysis
├── input/
│   ├── __init__.py
│   ├── text.py             # Text input handlers
│   ├── audio.py            # Audio input handlers
│   ├── image.py            # Image input handlers
│   └── video.py            # Video input handlers
├── output/
│   ├── __init__.py
│   ├── formatters.py       # Output formatters
│   ├── exporters.py        # Data exporters
│   └── visualizers.py      # Visualization tools
└── utils/
    ├── __init__.py
    ├── validators.py       # Input validation
    ├── converters.py       # Data converters
    └── helpers.py          # Helper functions
```

## Component Interactions

### Request Flow

1. User initializes `Sentimatrix` with configuration
2. Input layer receives and validates data
3. Scraping layer fetches external data (if needed)
4. Analysis layer processes data through selected models
5. Output layer formats and returns results

### Provider Selection

```python
# Provider selection is automatic based on config
config = SentimatrixConfig(
    llm_provider="groq",      # or "openai", "anthropic", "local"
    scraper_provider="playwright",  # or "selenium", "api"
    model_provider="huggingface"    # or "local", "api"
)
```

## Data Flow

```
Input Data
    │
    ▼
┌─────────────────┐
│ Input Validator │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Cache Check     │────▶│ Return Cached   │
└────────┬────────┘     └─────────────────┘
         │ (miss)
         ▼
┌─────────────────┐
│ Scraper/Loader  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessor    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Analysis Engine │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processor  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Formatter│
└─────────────────┘
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Async | asyncio, aiohttp |
| ML Framework | PyTorch, Transformers |
| Web Scraping | Playwright, Selenium, BeautifulSoup |
| API Framework | FastAPI (optional server mode) |
| Configuration | Pydantic, YAML |
| Caching | Redis, SQLite |
| Logging | structlog |
| Testing | pytest, pytest-asyncio |

## Scalability Considerations

### Horizontal Scaling
- Stateless design for easy replication
- Redis for distributed caching
- Message queue support for batch processing

### Vertical Scaling
- GPU acceleration for model inference
- Connection pooling for API calls
- Memory-efficient batch processing

## Security Considerations

- API key encryption at rest
- No credential logging
- Input sanitization
- Rate limiting support
- Proxy rotation for scraping
