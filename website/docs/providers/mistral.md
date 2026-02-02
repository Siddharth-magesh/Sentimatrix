---
title: Mistral AI
description: Integration with Mistral AI models
---

# Mistral AI

Mistral AI provides high-performance open-weight models with excellent multilingual capabilities.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="mistral",
        model="mistral-large-latest",
        api_key="your-mistral-key"  # Or set MISTRAL_API_KEY
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `mistral-large-latest` | 128K | Complex reasoning, analysis |
| `mistral-medium-latest` | 32K | Balanced performance |
| `mistral-small-latest` | 32K | Fast, cost-effective |
| `open-mistral-7b` | 32K | Lightweight tasks |
| `open-mixtral-8x7b` | 32K | MoE architecture |
| `open-mixtral-8x22b` | 64K | High performance MoE |
| `codestral-latest` | 32K | Code-focused tasks |

## Configuration

```python
LLMConfig(
    provider="mistral",
    model="mistral-large-latest",
    api_key="your-key",           # Or MISTRAL_API_KEY env var
    temperature=0.7,              # 0.0-1.0
    max_tokens=4096,
    top_p=1.0,
    timeout=30,
)
```

## Environment Variables

```bash
export MISTRAL_API_KEY="your-mistral-api-key"
```

## Features

- **Multilingual**: Excellent support for European languages
- **Function Calling**: Native tool/function support
- **JSON Mode**: Structured output generation
- **Streaming**: Real-time response streaming

## Pricing

| Model | Input | Output |
|-------|-------|--------|
| mistral-large | $2/1M | $6/1M |
| mistral-medium | $2.7/1M | $8.1/1M |
| mistral-small | $0.2/1M | $0.6/1M |

## Example: Multilingual Analysis

```python
async with Sentimatrix(config) as sm:
    # Mistral excels at multilingual content
    reviews_fr = ["Excellent produit!", "Pas terrible..."]
    reviews_de = ["Tolles Produkt!", "Nicht empfehlenswert"]

    results_fr = await sm.analyze_batch(reviews_fr)
    results_de = await sm.analyze_batch(reviews_de)
```

## Related

- [Provider Overview](index.md)
- [OpenAI](openai.md)
- [Anthropic](anthropic.md)
- [Selection Guide](selection-guide.md)
