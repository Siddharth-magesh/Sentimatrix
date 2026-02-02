---
title: Google Gemini
description: Use Google's Gemini models with Sentimatrix
---

# Google Gemini

Google Gemini offers multimodal capabilities with the industry's largest context window (up to 2M tokens).

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Context** | Up to 2M tokens |
| **Free Tier** | Yes |
| **Models** | Gemini 2.0, 1.5 Pro, 1.5 Flash |
| **Streaming** | :material-check: Supported |
| **Functions** | :material-check: Supported |
| **Vision** | :material-check: Supported |
| **Embeddings** | :material-check: Supported |

## Setup

### Get API Key

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Create or sign in with Google account
3. Get API key from API Keys section

### Configure

=== "Environment Variable"

    ```bash
    export GOOGLE_API_KEY="..."
    ```

=== "Python"

    ```python
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="gemini",
            api_key="...",
            model="gemini-1.5-pro"
        )
    )
    ```

=== "YAML"

    ```yaml
    llm:
      provider: gemini
      model: gemini-1.5-pro
    ```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `gemini-2.0-flash-exp` | 1M | Free* | Free* | Latest, experimental |
| `gemini-1.5-pro` | 2M | $1.25/1M | $5.00/1M | Long context |
| `gemini-1.5-flash` | 1M | $0.075/1M | $0.30/1M | Fast, cost-effective |
| `gemini-1.5-flash-8b` | 1M | $0.0375/1M | $0.15/1M | Ultra low cost |

*Free tier with rate limits

## Usage Examples

### Basic Usage

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="gemini",
        model="gemini-1.5-flash"
    )
)

async def main():
    async with Sentimatrix(config) as sm:
        summary = await sm.summarize_reviews(reviews)
        print(summary)

asyncio.run(main())
```

### Analyze Many Reviews (Long Context)

```python
# Gemini 1.5 Pro can handle 2M tokens - analyze thousands of reviews at once
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="gemini",
        model="gemini-1.5-pro"
    )
)

async with Sentimatrix(config) as sm:
    # Analyze 1000+ reviews in a single call
    insights = await sm.generate_insights(large_review_set)
```

### With Vision

```python
async with Sentimatrix(config) as sm:
    result = await sm.analyze_image(
        image_path="product.jpg",
        prompt="What sentiment does this product image convey?"
    )
```

## Configuration Options

```python
LLMConfig(
    provider="gemini",
    model="gemini-1.5-pro",

    # Generation settings
    temperature=0.7,
    max_tokens=8192,
    top_p=0.95,
    top_k=40,

    # Reliability
    timeout=60,
    max_retries=3,
)
```

## Best Practices

1. **Use Long Context for Bulk Analysis**
    - Gemini 1.5 Pro handles 2M tokens
    - Analyze entire review datasets in one call

2. **Start with Flash for Development**
    - Free tier available
    - Fast responses

3. **Use Pro for Production**
    - Better quality
    - Larger context

## Related

- [Provider Overview](index.md)
- [OpenAI](openai.md)
- [Anthropic](anthropic.md)
