---
title: DeepSeek
description: Use DeepSeek's cost-effective models with Sentimatrix
---

# DeepSeek

DeepSeek offers high-quality models at extremely competitive prices, making it ideal for cost-sensitive production workloads.

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Pricing** | Ultra-low cost ($0.07-0.27/1M tokens) |
| **Models** | DeepSeek V3, DeepSeek Coder, DeepSeek R1 |
| **Streaming** | :material-check: Supported |
| **Functions** | :material-check: Supported |
| **Embeddings** | :material-check: Supported |
| **JSON Mode** | :material-check: Supported |

## Setup

### Get API Key

1. Go to [platform.deepseek.com](https://platform.deepseek.com)
2. Create an account
3. Get API key from dashboard

### Configure

=== "Environment Variable"

    ```bash
    export DEEPSEEK_API_KEY="..."
    ```

=== "Python"

    ```python
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="deepseek",
            api_key="...",
            model="deepseek-chat"
        )
    )
    ```

=== "YAML"

    ```yaml
    llm:
      provider: deepseek
      model: deepseek-chat
    ```

## Available Models

| Model | Input Cost | Output Cost | Best For |
|-------|------------|-------------|----------|
| `deepseek-chat` | $0.07/1M | $0.27/1M | General chat |
| `deepseek-coder` | $0.07/1M | $0.27/1M | Code generation |
| `deepseek-reasoner` | $0.55/1M | $2.19/1M | Complex reasoning (R1) |

## Usage Examples

### Basic Usage

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="deepseek",
        model="deepseek-chat"
    )
)

async def main():
    async with Sentimatrix(config) as sm:
        # Extremely cost-effective summarization
        summary = await sm.summarize_reviews(reviews)
        print(summary)

asyncio.run(main())
```

### Cost Comparison

```python
# Analyze 10 million reviews
# DeepSeek: ~$5-10
# GPT-4o-mini: ~$10-15
# GPT-4o: ~$175+
# Claude 3.5: ~$250+

config = SentimatrixConfig(
    llm=LLMConfig(provider="deepseek", model="deepseek-chat")
)
```

## Configuration Options

```python
LLMConfig(
    provider="deepseek",
    model="deepseek-chat",

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

1. **Use for High-Volume Workloads**
    - Best cost-to-quality ratio
    - Great for production at scale

2. **Use deepseek-reasoner for Complex Tasks**
    - R1 model for reasoning
    - Higher cost but better quality

3. **Combine with Fallbacks**
    - Use DeepSeek as primary
    - Fall back to Groq/OpenAI

## Related

- [Provider Overview](index.md)
- [Groq](groq.md) - Another cost-effective option
- [Selection Guide](selection-guide.md)
