# Sentimatrix Changelog

> **Note:** The canonical changelog is located at [/CHANGELOG.md](../../CHANGELOG.md).
> This file provides a summary of major versions.

---

## Version 0.2.0 (Current Development)

### Summary

Sentimatrix V2 is a complete rewrite with:
- **19 LLM Providers** - Cloud, inference, local, and enterprise options
- **15 Scraper Integrations** - Platform-specific and commercial API scrapers
- **Full Sentiment Pipeline** - Analysis, emotions, multi-modal, exports
- **282+ Tests** - Comprehensive test coverage

### Key Features

| Category | Components |
|----------|------------|
| LLM Providers | OpenAI, Anthropic, Groq, Gemini, Mistral, Cohere, Together, Fireworks, OpenRouter, Cerebras, DeepSeek, Ollama, LM Studio, vLLM, llama.cpp, ExLlamaV2, Azure OpenAI, Bedrock |
| Platform Scrapers | Amazon, Steam, YouTube, Reddit, IMDB, Yelp, Trustpilot, Google Reviews |
| Commercial APIs | ScraperAPI, Apify, Bright Data, Oxylabs, Zyte, ScrapingBee, ScrapingAnt |
| Analysis | Sentiment (3/5 class), Emotions (28 GoEmotions), Multi-modal (audio/image/video) |
| Output | JSON, CSV, Excel, HTML, Markdown, Charts (matplotlib) |

### Breaking Changes from V1

- All APIs are now async (use `await`)
- Configuration via Pydantic models (not dicts)
- New import paths (`sentimatrix` not `Sentimatrix.sentiment_generation`)

### Migration Guide

```python
# Old (V1)
from Sentimatrix.sentiment_generation import SentConfig
sent = SentConfig(Use_Groq_API=True, Groq_API="gsk_...")
result = sent.get_Quick_sentiment("Great!")

# New (V2)
from sentimatrix import Sentimatrix, LLMConfig
config = LLMConfig(provider="groq", api_key="gsk_...")
async with Sentimatrix(llm_config=config) as sm:
    result = await sm.analyze_sentiment("Great!")
```

---

## Version 0.1.x (Previous)

### Features
- Basic sentiment analysis
- Limited LLM support (Groq, Gemini, Ollama)
- Basic web scraping (7 platforms)
- Synchronous API

### Limitations
- No async support
- Limited error handling
- Basic configuration
- Minimal testing

---

## Full Changelog

See [/CHANGELOG.md](../../CHANGELOG.md) for detailed stage-by-stage changes.
