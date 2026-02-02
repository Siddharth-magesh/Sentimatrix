---
title: LM Studio
description: Integration with LM Studio local inference
---

# LM Studio

LM Studio provides an easy-to-use desktop application for running LLMs locally with an OpenAI-compatible API.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="lmstudio",
        model="local-model",  # Model loaded in LM Studio
        api_base="http://localhost:1234/v1"
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Setup

1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai)
2. Download a model (e.g., Llama 3, Mistral, Phi)
3. Start the local server (default: port 1234)
4. Configure Sentimatrix to use the local endpoint

## Configuration

```python
LLMConfig(
    provider="lmstudio",
    model="local-model",
    api_base="http://localhost:1234/v1",  # LM Studio server
    temperature=0.7,
    max_tokens=4096,
    timeout=60,  # Local inference may be slower
)
```

## Recommended Models

| Model | Size | RAM Required |
|-------|------|--------------|
| Llama 3.2 3B | 3B | 4GB |
| Phi-3 Mini | 3.8B | 4GB |
| Mistral 7B | 7B | 8GB |
| Llama 3.1 8B | 8B | 8GB |
| Llama 3.1 70B | 70B | 48GB+ |

## Features

- **No API Key Required**: Completely local
- **Privacy**: Data never leaves your machine
- **Free**: No usage costs
- **OpenAI Compatible**: Standard API format

## System Requirements

| Model Size | RAM | GPU VRAM |
|------------|-----|----------|
| 3B | 4GB | 4GB |
| 7-8B | 8GB | 6GB |
| 13B | 16GB | 10GB |
| 70B | 48GB | 48GB |

## Example: Offline Analysis

```python
# Perfect for sensitive data that can't leave your network
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="lmstudio",
        api_base="http://localhost:1234/v1"
    )
)

async with Sentimatrix(config) as sm:
    # All processing happens locally
    sensitive_reviews = load_private_data()
    results = await sm.analyze_batch(sensitive_reviews)
```

## Related

- [Provider Overview](index.md)
- [Ollama](ollama.md)
- [vLLM](vllm.md)
- [Selection Guide](selection-guide.md)
