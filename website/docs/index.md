---
title: Sentimatrix - Advanced Sentiment Analysis Toolkit
description: Multi-provider LLM support, web scraping, and comprehensive sentiment analysis for Python
hide:
  - navigation
  - toc
---

<div class="hero">
  <h1>Sentimatrix</h1>
  <p class="tagline">Advanced sentiment analysis toolkit with multi-provider LLM support</p>
</div>

<div class="actions">
  <a href="getting-started/quickstart/" class="action-button primary">
    Get Started
  </a>
  <a href="https://github.com/sentimatrix/sentimatrix" class="action-button secondary">
    View on GitHub
  </a>
</div>

<div class="stats">
  <div class="stat">
    <div class="stat-number" data-suffix="+">19</div>
    <div class="stat-label">LLM Providers</div>
  </div>
  <div class="stat">
    <div class="stat-number" data-suffix="+">8</div>
    <div class="stat-label">Platform Scrapers</div>
  </div>
  <div class="stat">
    <div class="stat-number">7</div>
    <div class="stat-label">Commercial APIs</div>
  </div>
  <div class="stat">
    <div class="stat-number" data-suffix="+">280</div>
    <div class="stat-label">Tests</div>
  </div>
</div>

---

## What is Sentimatrix?

Sentimatrix is a production-ready Python library for **sentiment analysis**, **emotion detection**, and **review aggregation**. It combines the power of multiple LLM providers with robust web scraping capabilities to deliver comprehensive text analysis solutions.

```bash
pip install sentimatrix
```

---

## Quick Example

```python
import asyncio
from sentimatrix import Sentimatrix

async def main():
    async with Sentimatrix() as sm:
        # Analyze sentiment
        result = await sm.analyze("This product exceeded my expectations!")
        print(f"Sentiment: {result.sentiment}")  # positive
        print(f"Confidence: {result.confidence:.2%}")  # 94.32%

        # Detect emotions
        emotions = await sm.detect_emotions("I'm thrilled about this purchase!")
        print(f"Primary: {emotions.primary}")  # joy
        print(f"All: {emotions.scores}")  # {'joy': 0.89, 'surprise': 0.12, ...}

asyncio.run(main())
```

---

## Core Features

<div class="grid">

<div class="card">
<h3>:material-brain: Sentiment Analysis</h3>
<p>Multiple analysis modes including quick sentiment, structured analysis, aspect-based, comparative, and temporal analysis with domain-specific support.</p>
</div>

<div class="card">
<h3>:material-emoticon: Emotion Detection</h3>
<p>Detect emotions using Ekman's 6 basic emotions, GoEmotions' 28 classes, or Plutchik's wheel. Supports intensity analysis and emotion timelines.</p>
</div>

<div class="card">
<h3>:material-api: 19 LLM Providers</h3>
<p>Seamless integration with OpenAI, Anthropic, Google, Groq, Mistral, Cohere, Together, Fireworks, Ollama, vLLM, and more.</p>
</div>

<div class="card">
<h3>:material-web: Web Scraping</h3>
<p>Built-in scrapers for Amazon, Steam, YouTube, Reddit, IMDB, Yelp, Trustpilot, and Google Reviews with anti-detection measures.</p>
</div>

<div class="card">
<h3>:material-cloud: Commercial APIs</h3>
<p>Integration with ScraperAPI, Apify, Bright Data, Oxylabs, Zyte, ScrapingBee, and ScrapingAnt for enterprise-scale scraping.</p>
</div>

<div class="card">
<h3>:material-cog: Fully Async</h3>
<p>Built from the ground up with async/await support for high-performance concurrent operations.</p>
</div>

</div>

---

## LLM Providers

Sentimatrix supports a wide range of LLM providers for enhanced analysis:

<div class="grid">

<div class="card">
<h3>Cloud Providers</h3>
<p>
<span class="provider-badge cloud">OpenAI</span>
<span class="provider-badge cloud">Anthropic</span>
<span class="provider-badge cloud">Google</span>
<span class="provider-badge cloud">Mistral</span>
<span class="provider-badge cloud">Cohere</span>
<span class="provider-badge cloud">Groq</span>
</p>
</div>

<div class="card">
<h3>Inference Providers</h3>
<p>
<span class="provider-badge inference">Together</span>
<span class="provider-badge inference">Fireworks</span>
<span class="provider-badge inference">OpenRouter</span>
<span class="provider-badge inference">Cerebras</span>
<span class="provider-badge inference">DeepSeek</span>
</p>
</div>

<div class="card">
<h3>Local Providers</h3>
<p>
<span class="provider-badge local">Ollama</span>
<span class="provider-badge local">LM Studio</span>
<span class="provider-badge local">vLLM</span>
<span class="provider-badge local">llama.cpp</span>
<span class="provider-badge local">ExLlamaV2</span>
</p>
</div>

<div class="card">
<h3>Enterprise</h3>
<p>
<span class="provider-badge enterprise">Azure OpenAI</span>
<span class="provider-badge enterprise">AWS Bedrock</span>
</p>
</div>

</div>

---

## Platform Scrapers

Collect reviews and feedback from popular platforms:

| Platform | Type | Authentication | Rate Limit |
|----------|------|----------------|------------|
| **Amazon** | E-commerce | None | 10 req/min |
| **Steam** | Gaming | None | 20 req/min |
| **YouTube** | Video | API Key | 100 req/min |
| **Reddit** | Social | OAuth | 60 req/min |
| **IMDB** | Entertainment | None | 15 req/min |
| **Yelp** | Reviews | API Key | 50 req/min |
| **Trustpilot** | Reviews | None | 10 req/min |
| **Google Reviews** | Reviews | None | 5 req/min |

---

## Full Pipeline Example

Scrape reviews, analyze sentiment, and generate insights:

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

async def main():
    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            api_key="your-groq-key",
            model="llama-3.3-70b-versatile"
        )
    )

    async with Sentimatrix(config) as sm:
        # Scrape reviews from Steam
        reviews = await sm.scrape_reviews(
            url="https://store.steampowered.com/app/1245620/ELDEN_RING/",
            platform="steam",
            max_reviews=100
        )

        # Analyze all reviews
        results = await sm.analyze_batch([r.text for r in reviews])

        # Get distribution
        positive = sum(1 for r in results if r.sentiment == "positive")
        negative = sum(1 for r in results if r.sentiment == "negative")
        neutral = sum(1 for r in results if r.sentiment == "neutral")

        print(f"Positive: {positive}, Negative: {negative}, Neutral: {neutral}")

        # Generate LLM summary
        summary = await sm.summarize_reviews(reviews)
        print(f"\nSummary:\n{summary}")

asyncio.run(main())
```

---

## Installation

=== "pip"

    ```bash
    pip install sentimatrix
    ```

=== "pip (with extras)"

    ```bash
    # LLM providers
    pip install sentimatrix[llm]

    # Web scraping (includes Playwright)
    pip install sentimatrix[scraping]

    # Everything
    pip install sentimatrix[all]
    ```

=== "Poetry"

    ```bash
    poetry add sentimatrix
    ```

=== "uv"

    ```bash
    uv add sentimatrix
    ```

---

## Why Sentimatrix?

<div class="grid">

<div class="card">
<h3>:material-speedometer: Performance</h3>
<p>Async-first architecture with connection pooling, caching, and batch processing for maximum throughput.</p>
</div>

<div class="card">
<h3>:material-shield-check: Reliability</h3>
<p>Built-in retry logic, rate limiting, and fallback providers ensure your analysis pipelines stay running.</p>
</div>

<div class="card">
<h3>:material-puzzle: Extensibility</h3>
<p>Plugin architecture makes it easy to add new providers, scrapers, and analysis methods.</p>
</div>

<div class="card">
<h3>:material-test-tube: Well Tested</h3>
<p>280+ tests covering unit, integration, and end-to-end scenarios with comprehensive mocking.</p>
</div>

</div>

---

## Getting Help

- **Documentation**: You're here! Explore the sidebar for detailed guides.
- **GitHub Issues**: [Report bugs or request features](https://github.com/sentimatrix/sentimatrix/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/sentimatrix/sentimatrix/discussions)

---

<div style="text-align: center; margin-top: 3rem;">
  <a href="getting-started/quickstart/" class="action-button primary">
    Get Started with Sentimatrix
  </a>
</div>
