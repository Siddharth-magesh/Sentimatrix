---
title: Cerebras
description: Integration with Cerebras ultra-fast inference
---

# Cerebras

Cerebras provides the fastest LLM inference available, powered by their custom Wafer-Scale Engine chips.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="cerebras",
        model="llama3.1-70b",
        api_key="your-cerebras-key"  # Or set CEREBRAS_API_KEY
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Available Models

| Model | Context | Speed |
|-------|---------|-------|
| `llama3.1-70b` | 128K | 2000+ tokens/sec |
| `llama3.1-8b` | 128K | 4000+ tokens/sec |

## Configuration

```python
LLMConfig(
    provider="cerebras",
    model="llama3.1-70b",
    api_key="your-key",           # Or CEREBRAS_API_KEY env var
    temperature=0.7,
    max_tokens=4096,
    timeout=30,
)
```

## Environment Variables

```bash
export CEREBRAS_API_KEY="your-cerebras-api-key"
```

## Features

- **Ultra-Fast**: 10-20x faster than GPU inference
- **Low Latency**: Sub-100ms time-to-first-token
- **High Throughput**: 2000+ tokens/second
- **OpenAI Compatible**: Standard API format

## Performance

| Metric | Cerebras | GPU Cloud |
|--------|----------|-----------|
| Tokens/sec | 2000+ | 100-200 |
| Time to first token | <100ms | 500ms+ |
| Latency (70B) | ~2s | 20-30s |

## Use Cases

- **Real-time Applications**: Chat, live analysis
- **High-Volume Processing**: Batch analysis at scale
- **Interactive UX**: Instant responses

## Example: Speed Test

```python
import time

async with Sentimatrix(config) as sm:
    start = time.time()

    # Process 100 reviews
    results = await sm.analyze_batch(reviews[:100])

    elapsed = time.time() - start
    print(f"Processed 100 reviews in {elapsed:.2f}s")
    print(f"Rate: {100/elapsed:.1f} reviews/sec")
```

## Related

- [Provider Overview](index.md)
- [Groq](groq.md)
- [Selection Guide](selection-guide.md)
