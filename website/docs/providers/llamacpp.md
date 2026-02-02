---
title: llama.cpp
description: Integration with llama.cpp server
---

# llama.cpp

llama.cpp provides highly optimized CPU and GPU inference for GGUF quantized models.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="llamacpp",
        model="llama-3.1-8b",
        api_base="http://localhost:8080/v1"
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Setup

### Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make -j
```

### Start Server

```bash
./llama-server \
    -m models/llama-3.1-8b-instruct-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080
```

## Configuration

```python
LLMConfig(
    provider="llamacpp",
    model="llama-3.1-8b",
    api_base="http://localhost:8080/v1",
    temperature=0.7,
    max_tokens=4096,
    timeout=120,  # CPU inference can be slower
)
```

## Features

- **CPU Optimized**: AVX, AVX2, AVX-512 support
- **Quantization**: 2-8 bit quantization
- **Metal Support**: Apple Silicon acceleration
- **CUDA/ROCm**: GPU acceleration
- **Low Memory**: Run 70B on 32GB RAM

## Quantization Levels

| Format | Size Reduction | Quality |
|--------|---------------|---------|
| Q8_0 | 50% | Excellent |
| Q6_K | 60% | Very Good |
| Q5_K_M | 65% | Good |
| Q4_K_M | 75% | Good |
| Q3_K_M | 80% | Acceptable |
| Q2_K | 87% | Lower |

## Recommended Models (GGUF)

| Model | Q4_K_M Size | RAM Required |
|-------|-------------|--------------|
| Llama 3.2 3B | 2GB | 4GB |
| Llama 3.1 8B | 5GB | 8GB |
| Mistral 7B | 4GB | 8GB |
| Llama 3.1 70B | 40GB | 48GB |

## Example: CPU-Only Deployment

```python
# Perfect for servers without GPUs
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="llamacpp",
        api_base="http://localhost:8080/v1"
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze("Great product!")
```

## Server Options

```bash
./llama-server \
    -m model.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \           # Context length
    -ngl 35 \           # GPU layers (0 for CPU-only)
    --threads 8 \       # CPU threads
    --parallel 4        # Concurrent requests
```

## Related

- [Provider Overview](index.md)
- [Ollama](ollama.md)
- [ExLlamaV2](exllamav2.md)
- [Selection Guide](selection-guide.md)
