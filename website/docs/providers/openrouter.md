---
title: OpenRouter
description: Integration with OpenRouter unified API gateway
---

# OpenRouter

OpenRouter provides a unified API to access 100+ models from multiple providers through a single endpoint.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="openrouter",
        model="anthropic/claude-3.5-sonnet",
        api_key="your-openrouter-key"  # Or set OPENROUTER_API_KEY
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Available Models (Examples)

| Model | Provider | Context |
|-------|----------|---------|
| `anthropic/claude-3.5-sonnet` | Anthropic | 200K |
| `openai/gpt-4o` | OpenAI | 128K |
| `google/gemini-pro-1.5` | Google | 1M |
| `meta-llama/llama-3.1-405b-instruct` | Meta | 128K |
| `mistralai/mistral-large` | Mistral | 128K |
| `perplexity/llama-3.1-sonar-large-128k-online` | Perplexity | 128K |

## Configuration

```python
LLMConfig(
    provider="openrouter",
    model="anthropic/claude-3.5-sonnet",
    api_key="your-key",           # Or OPENROUTER_API_KEY env var
    temperature=0.7,
    max_tokens=4096,
    timeout=30,
)
```

## Environment Variables

```bash
export OPENROUTER_API_KEY="your-openrouter-api-key"
```

## Features

- **100+ Models**: Access all major providers
- **Unified Billing**: Single payment method
- **Automatic Fallback**: Switch providers on failure
- **Usage Tracking**: Detailed analytics

## Benefits

1. **Single API Key**: Access OpenAI, Anthropic, Google, and more
2. **Cost Optimization**: Compare pricing across providers
3. **No Rate Limits**: Aggregated capacity
4. **Model Discovery**: Try new models easily

## Example: Provider Comparison

```python
models = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "google/gemini-pro-1.5",
]

for model in models:
    config = SentimatrixConfig(
        llm=LLMConfig(provider="openrouter", model=model)
    )
    async with Sentimatrix(config) as sm:
        result = await sm.summarize_reviews(reviews)
        print(f"{model}: {result[:100]}...")
```

## Related

- [Provider Overview](index.md)
- [OpenAI](openai.md)
- [Anthropic](anthropic.md)
- [Selection Guide](selection-guide.md)
