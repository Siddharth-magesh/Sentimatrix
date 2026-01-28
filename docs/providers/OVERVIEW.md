# Sentimatrix V2 - LLM Providers Overview

## Architecture

V2 implements a unified provider interface that abstracts away differences between LLM providers. This allows seamless switching between providers without code changes.

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM PROVIDER MANAGER                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              BASE LLM PROVIDER                       │   │
│  │  - generate(prompt) -> str                          │   │
│  │  - generate_stream(prompt) -> AsyncIterator[str]    │   │
│  │  - embed(text) -> List[float]                       │   │
│  │  - supports_vision() -> bool                        │   │
│  │  - supports_function_calling() -> bool              │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│         ┌─────────────────┼─────────────────┐              │
│         ▼                 ▼                 ▼              │
│  ┌───────────┐     ┌───────────┐     ┌───────────┐        │
│  │  CLOUD    │     │  LOCAL    │     │ SPECIALIZED│        │
│  │ PROVIDERS │     │ PROVIDERS │     │ PROVIDERS  │        │
│  └───────────┘     └───────────┘     └───────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Provider Categories

### 1. Major Cloud Providers

| Provider | Models | Vision | Functions | Streaming |
|----------|--------|--------|-----------|-----------|
| OpenAI | GPT-4o, GPT-4, GPT-3.5 | Yes | Yes | Yes |
| Anthropic | Claude 3.5, Claude 3 | Yes | Yes | Yes |
| Google | Gemini 2.0, 1.5 | Yes | Yes | Yes |
| AWS Bedrock | Multiple | Varies | Varies | Yes |
| Azure OpenAI | GPT-4o, GPT-4 | Yes | Yes | Yes |

### 2. Open-Source Platforms

| Provider | Models | Pricing | Best For |
|----------|--------|---------|----------|
| Together AI | 200+ models | Pay-per-token | Variety |
| Fireworks AI | OSS models | Pay-per-token | Speed |
| Replicate | OSS models | Per-second | Flexibility |
| Hugging Face | 400K+ models | Free/Pro | Research |

### 3. Specialized Inference

| Provider | Speed | Specialty |
|----------|-------|-----------|
| Groq | 750 tok/s | Fastest cloud |
| Cerebras | 1800 tok/s | Fastest overall |
| SambaNova | 580 tok/s | Large models |

### 4. Local Inference

| Solution | Interface | Best For |
|----------|-----------|----------|
| Ollama | HTTP API | Ease of use |
| vLLM | HTTP API | Production |
| llama.cpp | CLI/API | Portability |
| LM Studio | GUI | Desktop |

### 5. Regional Providers

| Provider | Region | Models |
|----------|--------|--------|
| Mistral AI | EU | Mistral, Mixtral |
| DeepSeek | China | DeepSeek-R1 |
| Alibaba Qwen | China | Qwen series |
| Baidu ERNIE | China | ERNIE series |

---

## Provider Selection Matrix

### By Use Case

| Use Case | Recommended | Reason |
|----------|-------------|--------|
| Production (general) | OpenAI, Anthropic | Reliability |
| Cost-sensitive | Groq, DeepSeek | Low pricing |
| Speed-critical | Groq, Cerebras | Fastest |
| Privacy-required | Ollama, vLLM | Local |
| Vision tasks | GPT-4o, Claude 3 | Quality |
| Long context | Gemini, Claude | 1M+ tokens |
| Reasoning | Claude 3.5, o1 | Accuracy |

### By Budget

| Budget | Provider | Notes |
|--------|----------|-------|
| Free | Ollama, Gemini Flash | Limited/local |
| < $10/mo | Groq, DeepSeek | Free tiers |
| $10-100/mo | OpenAI, Anthropic | Standard use |
| $100-1000/mo | Multiple providers | High volume |
| > $1000/mo | Enterprise tiers | Dedicated |

---

## Pricing Comparison (per 1M tokens)

### Input Tokens

| Provider | Model | Price |
|----------|-------|-------|
| DeepSeek | V3 | $0.07 |
| Gemini | Flash | $0.075 |
| Groq | Llama 3.1 70B | $0.59 |
| OpenAI | GPT-4o-mini | $0.15 |
| OpenAI | GPT-4o | $2.50 |
| Anthropic | Sonnet 3.5 | $3.00 |
| Anthropic | Opus 3 | $15.00 |

### Output Tokens

| Provider | Model | Price |
|----------|-------|-------|
| DeepSeek | V3 | $0.27 |
| Gemini | Flash | $0.30 |
| Groq | Llama 3.1 70B | $0.79 |
| OpenAI | GPT-4o-mini | $0.60 |
| OpenAI | GPT-4o | $10.00 |
| Anthropic | Sonnet 3.5 | $15.00 |
| Anthropic | Opus 3 | $75.00 |

---

## Configuration

### Global LLM Config

```yaml
llm:
  default_provider: "openai"
  fallback_providers:
    - "anthropic"
    - "groq"

  timeout: 30
  max_retries: 3
  retry_delay: 1.0

  providers:
    openai:
      api_key: "${OPENAI_API_KEY}"
      model: "gpt-4o-mini"
      temperature: 0.7
      max_tokens: 1024

    anthropic:
      api_key: "${ANTHROPIC_API_KEY}"
      model: "claude-3-5-sonnet-20241022"
      max_tokens: 1024

    groq:
      api_key: "${GROQ_API_KEY}"
      model: "llama-3.1-70b-versatile"

    ollama:
      base_url: "http://localhost:11434"
      model: "llama3.1"
```

---

## Feature Support Matrix

| Provider | Streaming | Functions | Vision | Embeddings | JSON Mode |
|----------|-----------|-----------|--------|------------|-----------|
| OpenAI | Yes | Yes | Yes | Yes | Yes |
| Anthropic | Yes | Yes | Yes | No | Yes |
| Google | Yes | Yes | Yes | Yes | Yes |
| Groq | Yes | Yes | No | No | Yes |
| Mistral | Yes | Yes | No | Yes | Yes |
| Ollama | Yes | Partial | Yes* | Yes | Yes |
| Together | Yes | Yes | Yes* | Yes | Yes |
| DeepSeek | Yes | Yes | No | Yes | Yes |

*Depends on model

---

## Unified Interface

```python
from sentimatrix.providers.llm import get_provider

# Get provider by name
provider = get_provider("openai", api_key="...")

# All providers support the same interface
response = await provider.generate(
    prompt="Analyze sentiment: This product is amazing!",
    system_prompt="You are a sentiment analysis expert.",
    temperature=0.7,
    max_tokens=500
)

# Streaming
async for chunk in provider.generate_stream(prompt):
    print(chunk, end="")

# Embeddings (if supported)
if provider.supports_embeddings():
    vector = await provider.embed("Some text to embed")
```

---

## Fallback Strategy

```python
# Automatic fallback on failure
llm_manager = LLMManager(config)
llm_manager.set_fallback_chain(["openai", "anthropic", "groq"])

# Will try openai first, then anthropic, then groq
response = await llm_manager.generate(prompt)
```

---

## Rate Limiting

Each provider has different rate limits:

| Provider | Requests/min | Tokens/min |
|----------|--------------|------------|
| OpenAI (Tier 1) | 500 | 30,000 |
| OpenAI (Tier 4) | 10,000 | 800,000 |
| Anthropic | 1,000 | 80,000 |
| Groq | 30 | 6,000 |
| Gemini | 60 | 1,000,000 |

V2 includes built-in rate limiting to respect these limits.
