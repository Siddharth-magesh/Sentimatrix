# Sentimatrix V2 - LLM Providers Overview

## Architecture

Sentimatrix V2 implements a unified provider interface with 19 LLM providers. This allows seamless switching between providers without code changes.

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM PROVIDER MANAGER                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                 BASE LLM PROVIDER                        │    │
│  │  - generate(prompt) -> LLMResponse                       │    │
│  │  - generate_stream(prompt) -> AsyncIterator[str]        │    │
│  │  - embed(text) -> List[float]                           │    │
│  │  - generate_with_functions(prompt, tools) -> Response   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                           │                                      │
│    ┌──────────────────────┼──────────────────────┐              │
│    ▼                      ▼                      ▼              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   CLOUD     │   │  INFERENCE  │   │   LOCAL     │           │
│  │  PROVIDERS  │   │  PROVIDERS  │   │  PROVIDERS  │           │
│  └─────────────┘   └─────────────┘   └─────────────┘           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## All 19 Implemented Providers

### Cloud Providers (6)

| Provider | Class | Models | Free Tier |
|----------|-------|--------|-----------|
| OpenAI | `OpenAIProvider` | GPT-4o, GPT-4o-mini, o1 | No |
| Anthropic | `AnthropicProvider` | Claude 3.5 Sonnet, Claude 3 | No |
| Google Gemini | `GeminiProvider` | Gemini 2.0 Flash, 1.5 Pro | Yes |
| Mistral | `MistralProvider` | Mistral 7B, 8x7B, Large | Yes |
| Cohere | `CohereProvider` | Command R, Command R+ | Yes |
| Groq | `GroqProvider` | LLaMA 3.3, Mixtral | Yes |

### Inference Providers (5)

| Provider | Class | Models | Speed |
|----------|-------|--------|-------|
| Together AI | `TogetherProvider` | 200+ models | Fast |
| Fireworks AI | `FireworksProvider` | OSS models | Very Fast |
| OpenRouter | `OpenRouterProvider` | All models | Varies |
| Cerebras | `CerebrasProvider` | LLaMA | Ultra Fast |
| DeepSeek | `DeepSeekProvider` | DeepSeek V3, R1 | Fast |

### Local Providers (6)

| Provider | Class | Interface | Best For |
|----------|-------|-----------|----------|
| Ollama | `OllamaProvider` | HTTP :11434 | Easy setup |
| LM Studio | `LMStudioProvider` | HTTP :1234 | Desktop GUI |
| vLLM | `vLLMProvider` | HTTP | Production |
| llama.cpp | `LlamaCppProvider` | HTTP | Portability |
| text-gen-webui | `TextGenProvider` | HTTP :5000 | Features |
| ExLlamaV2 | `ExLlamaV2Provider` | HTTP | Quantized |

### Enterprise (2)

| Provider | Class | Region | Notes |
|----------|-------|--------|-------|
| Azure OpenAI | `AzureOpenAIProvider` | Azure | Enterprise |
| AWS Bedrock | `BedrockProvider` | AWS | Multi-model |

---

## Quick Start

### Basic Usage

```python
import asyncio
from sentimatrix.providers.llm import GroqProvider
from sentimatrix.core.config import LLMConfig

async def main():
    config = LLMConfig(
        provider="groq",
        api_key="gsk_...",
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

    async with GroqProvider(config) as provider:
        response = await provider.generate(
            prompt="Analyze the sentiment: I love this product!",
            system_prompt="You are a sentiment analyst.",
        )

        print(f"Response: {response.content}")
        print(f"Tokens: {response.usage.total_tokens}")
        print(f"Time: {response.response_time_ms:.0f}ms")

asyncio.run(main())
```

### Using the Provider Manager

```python
from sentimatrix.providers.llm.manager import LLMProviderManager, LLMProvider

async def main():
    manager = LLMProviderManager(
        api_keys={
            "groq": "gsk_...",
            "openai": "sk-...",
        }
    )

    # Initialize primary provider
    await manager.initialize(LLMProvider.GROQ)

    # Generate with automatic fallback
    response = await manager.generate(
        prompt="Summarize this review...",
        fallback_providers=[LLMProvider.OPENAI],
    )

asyncio.run(main())
```

### Streaming

```python
async with GroqProvider(config) as provider:
    async for chunk in provider.generate_stream("Tell me a story..."):
        print(chunk, end="", flush=True)
```

### Function Calling

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_sentiment",
            "description": "Analyze sentiment of text",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number"},
                },
                "required": ["sentiment", "confidence"],
            },
        },
    }
]

response = await provider.generate_with_functions(
    prompt="This product is amazing!",
    functions=tools,
)
```

---

## Provider Selection Guide

### By Use Case

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Production | OpenAI, Anthropic | Reliability |
| Cost-sensitive | Groq, DeepSeek | Low/free pricing |
| Speed-critical | Groq, Cerebras | Fastest inference |
| Privacy-required | Ollama, vLLM | Local execution |
| Vision tasks | GPT-4o, Claude 3 | Quality |
| Long context | Gemini, Claude | 1M+ tokens |
| Reasoning | Claude 3.5, o1 | Accuracy |

### By Budget

| Budget | Provider | Notes |
|--------|----------|-------|
| Free | Groq, Gemini, Ollama | Limited quotas |
| < $10/mo | Groq, Mistral | Free tiers |
| $10-100/mo | OpenAI, Anthropic | Standard use |
| > $100/mo | Enterprise tiers | High volume |

---

## Pricing Comparison (per 1M tokens)

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| Groq | LLaMA 3.3 70B | Free | Free |
| DeepSeek | V3 | $0.07 | $0.27 |
| Gemini | 1.5 Flash | $0.075 | $0.30 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 |
| OpenAI | GPT-4o | $2.50 | $10.00 |
| Anthropic | Claude 3.5 Sonnet | $3.00 | $15.00 |
| Anthropic | Claude 3 Opus | $15.00 | $75.00 |

---

## Feature Support Matrix

| Provider | Streaming | Functions | Vision | Embeddings | JSON Mode |
|----------|-----------|-----------|--------|------------|-----------|
| OpenAI | Yes | Yes | Yes | Yes | Yes |
| Anthropic | Yes | Yes | Yes | No | Yes |
| Google | Yes | Yes | Yes | Yes | Yes |
| Groq | Yes | Yes | No | No | Yes |
| Mistral | Yes | Yes | No | Yes | Yes |
| Cohere | Yes | Yes | No | Yes | Yes |
| Together | Yes | Yes | Yes* | Yes | Yes |
| Fireworks | Yes | Yes | Yes* | Yes | Yes |
| DeepSeek | Yes | Yes | No | Yes | Yes |
| Ollama | Yes | Partial | Yes* | Yes | Yes |

*Depends on model

---

## Configuration

### YAML Config

```yaml
llm:
  default_provider: groq
  fallback_providers:
    - openai
    - anthropic

  timeout: 30
  max_retries: 3

  providers:
    groq:
      api_key: ${GROQ_API_KEY}
      model: llama-3.3-70b-versatile
      temperature: 0.7

    openai:
      api_key: ${OPENAI_API_KEY}
      model: gpt-4o-mini

    ollama:
      base_url: http://localhost:11434
      model: llama3.2
```

### Environment Variables

```bash
export GROQ_API_KEY="gsk_..."
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export MISTRAL_API_KEY="..."
export TOGETHER_API_KEY="..."
```

---

## Rate Limits

| Provider | Requests/min | Tokens/min | Free Tier |
|----------|--------------|------------|-----------|
| Groq | 30 | 6,000 | Yes |
| OpenAI (Tier 1) | 500 | 30,000 | No |
| Anthropic | 1,000 | 80,000 | No |
| Gemini | 60 | 1,000,000 | Yes |
| Mistral | 100 | 500,000 | Yes |

The provider manager includes built-in rate limiting and automatic retry with backoff.

---

## Related Documentation

- [Cloud Providers](./CLOUD_PROVIDERS.md) - OpenAI, Anthropic, Google details
- [Inference Providers](./INFERENCE_PROVIDERS.md) - Groq, Together, Fireworks details
- [Local Providers](./LOCAL_PROVIDERS.md) - Ollama, vLLM, llama.cpp details
- [Provider Manager](../api/REFERENCE.md) - Manager API reference
