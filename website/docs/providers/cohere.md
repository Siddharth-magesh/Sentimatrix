---
title: Cohere
description: Integration with Cohere's Command models
---

# Cohere

Cohere provides enterprise-focused NLP models with excellent RAG and embedding capabilities.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="cohere",
        model="command-r-plus",
        api_key="your-cohere-key"  # Or set COHERE_API_KEY
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `command-r-plus` | 128K | Complex tasks, RAG |
| `command-r` | 128K | Balanced performance |
| `command` | 4K | Legacy, fast |
| `command-light` | 4K | Lightweight tasks |

## Configuration

```python
LLMConfig(
    provider="cohere",
    model="command-r-plus",
    api_key="your-key",           # Or COHERE_API_KEY env var
    temperature=0.7,
    max_tokens=4096,
    timeout=30,
)
```

## Environment Variables

```bash
export COHERE_API_KEY="your-cohere-api-key"
```

## Features

- **RAG Optimized**: Built-in citation and grounding
- **Embeddings**: High-quality text embeddings
- **Reranking**: Document reranking capabilities
- **Multilingual**: 100+ language support

## Pricing

| Model | Input | Output |
|-------|-------|--------|
| command-r-plus | $2.50/1M | $10/1M |
| command-r | $0.15/1M | $0.60/1M |
| command | $1/1M | $2/1M |

## Example: With Citations

```python
async with Sentimatrix(config) as sm:
    # Cohere excels at grounded responses
    insights = await sm.generate_insights(
        reviews,
        include_citations=True
    )

    print(insights.summary)
    print(insights.citations)  # Source references
```

## Related

- [Provider Overview](index.md)
- [OpenAI](openai.md)
- [Selection Guide](selection-guide.md)
