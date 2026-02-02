---
title: Together AI
description: Integration with Together AI inference platform
---

# Together AI

Together AI provides fast inference for open-source models with competitive pricing.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="together",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        api_key="your-together-key"  # Or set TOGETHER_API_KEY
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Available Models

| Model | Context | Best For |
|-------|---------|----------|
| `meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo` | 128K | Highest quality |
| `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` | 128K | Balanced |
| `meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo` | 128K | Fast, economical |
| `mistralai/Mixtral-8x22B-Instruct-v0.1` | 64K | MoE performance |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 32K | Fast MoE |
| `Qwen/Qwen2-72B-Instruct` | 128K | Multilingual |
| `deepseek-ai/deepseek-llm-67b-chat` | 4K | Cost-effective |

## Configuration

```python
LLMConfig(
    provider="together",
    model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    api_key="your-key",           # Or TOGETHER_API_KEY env var
    temperature=0.7,
    max_tokens=4096,
    timeout=30,
)
```

## Environment Variables

```bash
export TOGETHER_API_KEY="your-together-api-key"
```

## Features

- **Wide Model Selection**: 100+ open-source models
- **Fast Inference**: Optimized serving infrastructure
- **Competitive Pricing**: Often cheaper than original providers
- **OpenAI Compatible**: Drop-in API compatibility

## Pricing (Examples)

| Model | Input | Output |
|-------|-------|--------|
| Llama 3.1 405B | $3.50/1M | $3.50/1M |
| Llama 3.1 70B | $0.88/1M | $0.88/1M |
| Llama 3.1 8B | $0.18/1M | $0.18/1M |
| Mixtral 8x22B | $1.20/1M | $1.20/1M |

## Example: Model Comparison

```python
models = [
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
]

for model in models:
    config = SentimatrixConfig(
        llm=LLMConfig(provider="together", model=model)
    )
    async with Sentimatrix(config) as sm:
        result = await sm.analyze("Great product!")
        print(f"{model}: {result.sentiment}")
```

## Related

- [Provider Overview](index.md)
- [Fireworks](fireworks.md)
- [Selection Guide](selection-guide.md)
