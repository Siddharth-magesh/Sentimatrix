---
title: Fireworks AI
description: Integration with Fireworks AI inference platform
---

# Fireworks AI

Fireworks AI provides the fastest inference speeds for open-source models with enterprise reliability.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="fireworks",
        model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        api_key="your-fireworks-key"  # Or set FIREWORKS_API_KEY
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `accounts/fireworks/models/llama-v3p1-405b-instruct` | 128K | Highest quality |
| `accounts/fireworks/models/llama-v3p1-70b-instruct` | 128K | Balanced |
| `accounts/fireworks/models/llama-v3p1-8b-instruct` | 128K | Fast |
| `accounts/fireworks/models/mixtral-8x22b-instruct` | 64K | MoE |
| `accounts/fireworks/models/qwen2-72b-instruct` | 128K | Multilingual |
| `accounts/fireworks/models/firefunction-v2` | 8K | Function calling |

## Configuration

```python
LLMConfig(
    provider="fireworks",
    model="accounts/fireworks/models/llama-v3p1-70b-instruct",
    api_key="your-key",           # Or FIREWORKS_API_KEY env var
    temperature=0.7,
    max_tokens=4096,
    timeout=30,
)
```

## Environment Variables

```bash
export FIREWORKS_API_KEY="your-fireworks-api-key"
```

## Features

- **Fastest Inference**: Industry-leading speed
- **Function Calling**: FireFunction models
- **Batch API**: High-throughput batch processing
- **OpenAI Compatible**: Drop-in replacement

## Pricing (Examples)

| Model | Input | Output |
|-------|-------|--------|
| Llama 3.1 405B | $3/1M | $3/1M |
| Llama 3.1 70B | $0.90/1M | $0.90/1M |
| Llama 3.1 8B | $0.20/1M | $0.20/1M |
| Mixtral 8x22B | $0.90/1M | $0.90/1M |

## Example: High-Throughput Processing

```python
async with Sentimatrix(config) as sm:
    # Fireworks excels at batch processing
    reviews = [...]  # 1000s of reviews

    results = await sm.analyze_batch(
        reviews,
        batch_size=100  # Large batches supported
    )
```

## Related

- [Provider Overview](index.md)
- [Together AI](together.md)
- [Groq](groq.md)
- [Selection Guide](selection-guide.md)
