---
title: OpenAI
description: Use OpenAI's GPT models with Sentimatrix
---

# OpenAI

OpenAI provides state-of-the-art language models including GPT-4o and the reasoning-focused o1 series.

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Quality** | State-of-the-art |
| **Free Tier** | No (pay-as-you-go) |
| **Models** | GPT-4o, GPT-4o-mini, o1 |
| **Streaming** | :material-check: Supported |
| **Functions** | :material-check: Supported |
| **Vision** | :material-check: Supported |
| **Embeddings** | :material-check: Supported |

## Setup

### Get API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Create an account and add payment method
3. Navigate to API Keys
4. Create a new secret key

### Configure

=== "Environment Variable"

    ```bash
    export OPENAI_API_KEY="sk-..."
    ```

=== "Python"

    ```python
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="openai",
            api_key="sk-...",
            model="gpt-4o-mini"
        )
    )
    ```

=== "YAML"

    ```yaml
    llm:
      provider: openai
      model: gpt-4o-mini
    ```

## Available Models

| Model | Context | Input Cost | Output Cost | Best For |
|-------|---------|------------|-------------|----------|
| `gpt-4o` | 128K | $2.50/1M | $10.00/1M | Best quality |
| `gpt-4o-mini` | 128K | $0.15/1M | $0.60/1M | Cost-effective |
| `o1` | 200K | $15.00/1M | $60.00/1M | Complex reasoning |
| `o1-mini` | 128K | $3.00/1M | $12.00/1M | Fast reasoning |
| `gpt-4-turbo` | 128K | $10.00/1M | $30.00/1M | Legacy |

## Usage Examples

### Basic Usage

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o-mini"
    )
)

async def main():
    async with Sentimatrix(config) as sm:
        summary = await sm.summarize_reviews(reviews)
        print(summary)

asyncio.run(main())
```

### With Vision (Analyze Images)

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o"  # Vision-capable model
    )
)

async with Sentimatrix(config) as sm:
    # Analyze product image
    result = await sm.analyze_image(
        image_path="product.jpg",
        prompt="Describe the sentiment conveyed by this product image"
    )
```

### Structured Output (JSON Mode)

```python
async with Sentimatrix(config) as sm:
    insights = await sm.generate_insights(
        reviews,
        output_format="json",
        schema={
            "pros": ["string"],
            "cons": ["string"],
            "rating": "number",
            "recommendation": "string"
        }
    )
```

### Function Calling

```python
async with Sentimatrix(config) as sm:
    # Automatic function calling for structured extraction
    analysis = await sm.extract_aspects(
        text="The battery life is great but the screen is too dim.",
        aspects=["battery", "screen", "price", "design"]
    )
    # Returns: {"battery": "positive", "screen": "negative", ...}
```

## Configuration Options

```python
LLMConfig(
    provider="openai",
    model="gpt-4o-mini",

    # API settings
    api_key="sk-...",
    organization="org-...",  # Optional
    base_url=None,           # For Azure/proxies

    # Generation settings
    temperature=0.7,
    max_tokens=4096,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0,

    # Reliability
    timeout=60,
    max_retries=3,

    # Features
    json_mode=False,         # Force JSON output
    seed=None,               # For reproducibility
)
```

## Rate Limits

OpenAI has tiered rate limits based on usage:

| Tier | RPM | TPM |
|------|-----|-----|
| Free | 3 | 40,000 |
| Tier 1 | 500 | 200,000 |
| Tier 2 | 5,000 | 2,000,000 |
| Tier 3+ | Higher | Higher |

### Handling Rate Limits

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4o-mini",
        rate_limit={
            "requests_per_minute": 450,
            "tokens_per_minute": 180000,
            "retry_on_rate_limit": True,
        }
    )
)
```

## Best Practices

1. **Choose the Right Model**
    - `gpt-4o-mini` for most tasks (best value)
    - `gpt-4o` for complex analysis
    - `o1` for multi-step reasoning

2. **Use JSON Mode for Structured Output**
    ```python
    LLMConfig(provider="openai", json_mode=True)
    ```

3. **Enable Caching**
    - OpenAI supports prompt caching for repeated requests
    - Reduces costs by up to 50%

4. **Monitor Costs**
    - Use usage tracking
    - Set spending limits in OpenAI dashboard

## Troubleshooting

??? question "Invalid API key"

    Verify your API key:

    ```python
    import openai
    client = openai.OpenAI(api_key="sk-...")
    models = client.models.list()
    ```

??? question "Rate limit exceeded"

    Implement backoff or reduce concurrency:

    ```python
    LLMConfig(
        rate_limit={"requests_per_minute": 100},
        max_retries=5,
        retry_delay=2.0
    )
    ```

??? question "Context length exceeded"

    Reduce input size or use a model with larger context:

    ```python
    # GPT-4o supports 128K context
    LLMConfig(model="gpt-4o")
    ```

## Related

- [Provider Overview](index.md)
- [Azure OpenAI](azure.md) - Enterprise deployment
- [Anthropic](anthropic.md) - Alternative premium provider
