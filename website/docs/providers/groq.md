---
title: Groq
description: Use Groq's ultra-fast inference with Sentimatrix
---

# Groq

Groq provides ultra-fast LLM inference using custom hardware (LPUs). It offers a generous free tier, making it perfect for getting started with Sentimatrix.

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Speed** | Ultra-fast (100+ tokens/s) |
| **Free Tier** | Yes (generous limits) |
| **Models** | LLaMA 3.3, Mixtral, Gemma |
| **Streaming** | :material-check: Supported |
| **Functions** | :material-check: Supported |
| **Vision** | :material-close: Not supported |

## Setup

### Get API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Create an account (free)
3. Navigate to API Keys
4. Create a new API key

### Configure

=== "Environment Variable"

    ```bash
    export GROQ_API_KEY="gsk_..."
    ```

=== "Python"

    ```python
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            api_key="gsk_...",
            model="llama-3.3-70b-versatile"
        )
    )
    ```

=== "YAML"

    ```yaml
    llm:
      provider: groq
      model: llama-3.3-70b-versatile
    ```

## Available Models

| Model | Context | Speed | Best For |
|-------|---------|-------|----------|
| `llama-3.3-70b-versatile` | 128K | Fast | General use, best quality |
| `llama-3.1-70b-versatile` | 128K | Fast | General use |
| `llama-3.1-8b-instant` | 128K | Ultra-fast | Quick tasks |
| `mixtral-8x7b-32768` | 32K | Fast | Code, reasoning |
| `gemma2-9b-it` | 8K | Ultra-fast | Lightweight tasks |

## Usage Examples

### Basic Summarization

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile"
    )
)

async def main():
    async with Sentimatrix(config) as sm:
        reviews = [
            {"text": "Amazing product, works perfectly!"},
            {"text": "Good value for the price."},
            {"text": "Shipping was slow but product is fine."},
        ]

        summary = await sm.summarize_reviews(reviews)
        print(summary)

asyncio.run(main())
```

### Generate Insights

```python
async with Sentimatrix(config) as sm:
    insights = await sm.generate_insights(reviews)

    print("Pros:")
    for pro in insights.pros:
        print(f"  + {pro}")

    print("\nCons:")
    for con in insights.cons:
        print(f"  - {con}")
```

### Streaming Responses

```python
async with Sentimatrix(config) as sm:
    async for chunk in sm.stream_summary(reviews):
        print(chunk, end="", flush=True)
```

## Rate Limits

Groq's free tier has generous limits:

| Model | Requests/min | Tokens/min | Tokens/day |
|-------|--------------|------------|------------|
| LLaMA 3.3 70B | 30 | 6,000 | 100,000 |
| LLaMA 3.1 8B | 30 | 20,000 | 500,000 |
| Mixtral 8x7B | 30 | 5,000 | 100,000 |

### Handling Rate Limits

Sentimatrix handles rate limits automatically:

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",
        model="llama-3.3-70b-versatile",
        rate_limit={
            "requests_per_minute": 25,  # Stay under limit
            "retry_on_rate_limit": True,
        }
    )
)
```

## Configuration Options

```python
LLMConfig(
    provider="groq",
    model="llama-3.3-70b-versatile",

    # Generation settings
    temperature=0.7,          # Creativity (0-2)
    max_tokens=4096,          # Max response length
    top_p=0.9,                # Nucleus sampling

    # Reliability
    timeout=30,               # Request timeout
    max_retries=3,            # Retry count
    retry_delay=1.0,          # Initial retry delay

    # Rate limiting
    rate_limit={
        "requests_per_minute": 25,
        "tokens_per_minute": 5000,
    }
)
```

## Best Practices

1. **Use the Right Model**
    - `llama-3.3-70b-versatile` for best quality
    - `llama-3.1-8b-instant` for speed-critical tasks

2. **Handle Rate Limits**
    - Implement exponential backoff
    - Cache responses when possible

3. **Optimize Prompts**
    - Keep prompts concise
    - Use structured output formats

## Troubleshooting

??? question "Rate limit exceeded"

    Reduce request frequency or upgrade to a paid plan:

    ```python
    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="groq",
            rate_limit={"requests_per_minute": 20}
        )
    )
    ```

??? question "Model not found"

    Check available models at [console.groq.com/docs/models](https://console.groq.com/docs/models)

??? question "Timeout errors"

    Increase timeout for large requests:

    ```python
    LLMConfig(provider="groq", timeout=60)
    ```

## Related

- [Provider Overview](index.md)
- [OpenAI](openai.md) - Alternative cloud provider
- [Ollama](ollama.md) - Local alternative
