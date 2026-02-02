---
title: Anthropic
description: Use Claude models with Sentimatrix
---

# Anthropic

Anthropic provides Claude, known for its strong reasoning capabilities, safety features, and long context windows up to 200K tokens.

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Quality** | State-of-the-art reasoning |
| **Context** | Up to 200K tokens |
| **Models** | Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku |
| **Streaming** | :material-check: Supported |
| **Functions** | :material-check: Supported |
| **Vision** | :material-check: Supported |
| **Batching** | :material-check: Supported |

## Setup

### Get API Key

1. Go to [console.anthropic.com](https://console.anthropic.com)
2. Create an account
3. Navigate to API Keys
4. Create a new API key

### Configure

=== "Environment Variable"

    ```bash
    export ANTHROPIC_API_KEY="sk-ant-..."
    ```

=== "Python"

    ```python
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="anthropic",
            api_key="sk-ant-...",
            model="claude-3-5-sonnet-20241022"
        )
    )
    ```

=== "YAML"

    ```yaml
    llm:
      provider: anthropic
      model: claude-3-5-sonnet-20241022
    ```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `claude-3-5-sonnet-20241022` | 200K | $3.00/1M | $15.00/1M | Best overall |
| `claude-3-opus-20240229` | 200K | $15.00/1M | $75.00/1M | Complex reasoning |
| `claude-3-sonnet-20240229` | 200K | $3.00/1M | $15.00/1M | Balanced |
| `claude-3-haiku-20240307` | 200K | $0.25/1M | $1.25/1M | Fast, cost-effective |

## Usage Examples

### Basic Usage

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022"
    )
)

async def main():
    async with Sentimatrix(config) as sm:
        insights = await sm.generate_insights(reviews)
        print(f"Pros: {insights.pros}")
        print(f"Cons: {insights.cons}")

asyncio.run(main())
```

### With Vision

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022"
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze_image(
        image_path="product.jpg",
        prompt="Analyze the sentiment conveyed by this product image"
    )
```

### Streaming

```python
async with Sentimatrix(config) as sm:
    async for chunk in sm.stream_summary(reviews):
        print(chunk, end="", flush=True)
```

## Configuration Options

```python
LLMConfig(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",

    # API settings
    api_key="sk-ant-...",

    # Generation settings
    temperature=0.7,
    max_tokens=4096,
    top_p=0.9,

    # Reliability
    timeout=60,
    max_retries=3,
)
```

## Best Practices

1. **Choose the Right Model**
    - `claude-3-5-sonnet` for best quality
    - `claude-3-haiku` for speed and cost

2. **Use Long Context Wisely**
    - Claude supports 200K tokens
    - Great for analyzing many reviews at once

3. **Leverage Vision**
    - Analyze product images alongside text

## Related

- [Provider Overview](index.md)
- [OpenAI](openai.md)
- [Groq](groq.md)
