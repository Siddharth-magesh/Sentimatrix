---
title: ExLlamaV2
description: Integration with ExLlamaV2 optimized inference
---

# ExLlamaV2

ExLlamaV2 provides extremely fast GPU inference for EXL2 quantized models with advanced memory optimization.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="exllamav2",
        model="llama-3.1-8b-exl2",
        api_base="http://localhost:5000/v1"
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Setup

### Install ExLlamaV2

```bash
pip install exllamav2
```

### Start Server (TabbyAPI)

```bash
pip install tabbyapi
python -m tabbyapi.main \
    --model-dir /path/to/model \
    --host 0.0.0.0 \
    --port 5000
```

## Configuration

```python
LLMConfig(
    provider="exllamav2",
    model="llama-3.1-8b-exl2",
    api_base="http://localhost:5000/v1",
    temperature=0.7,
    max_tokens=4096,
    timeout=60,
)
```

## Features

- **Fastest GPU Inference**: Optimized CUDA kernels
- **EXL2 Format**: Superior quantization quality
- **Dynamic Batching**: Efficient batch processing
- **Speculative Decoding**: Even faster generation
- **Flash Attention**: Memory-efficient attention

## EXL2 Quantization

| Bits per Weight | VRAM Savings | Quality |
|-----------------|--------------|---------|
| 8.0 bpw | 50% | Excellent |
| 6.0 bpw | 62% | Very Good |
| 5.0 bpw | 68% | Good |
| 4.0 bpw | 75% | Good |
| 3.0 bpw | 81% | Acceptable |

## Performance Comparison

| Engine | Tokens/sec (8B) | VRAM Usage |
|--------|-----------------|------------|
| ExLlamaV2 | 150+ | 6GB |
| vLLM | 100 | 8GB |
| Transformers | 30 | 16GB |

## Example: Maximum Performance

```python
# ExLlamaV2 for speed-critical applications
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="exllamav2",
        api_base="http://localhost:5000/v1"
    )
)

async with Sentimatrix(config) as sm:
    # Ultra-fast inference
    result = await sm.analyze("Great product!")
```

## Model Sources

- [Hugging Face EXL2 Models](https://huggingface.co/models?sort=trending&search=exl2)
- Popular quantizers: turboderp, TheBloke

## Related

- [Provider Overview](index.md)
- [vLLM](vllm.md)
- [llama.cpp](llamacpp.md)
- [Selection Guide](selection-guide.md)
