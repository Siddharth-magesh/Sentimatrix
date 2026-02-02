---
title: vLLM
description: Integration with vLLM high-performance inference server
---

# vLLM

vLLM is a high-throughput, memory-efficient inference engine for LLMs with PagedAttention technology.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="vllm",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_base="http://localhost:8000/v1"
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Setup

### Install vLLM

```bash
pip install vllm
```

### Start Server

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000
```

## Configuration

```python
LLMConfig(
    provider="vllm",
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    api_base="http://localhost:8000/v1",
    temperature=0.7,
    max_tokens=4096,
    timeout=60,
)
```

## Features

- **PagedAttention**: Efficient memory management
- **Continuous Batching**: High throughput
- **Tensor Parallelism**: Multi-GPU support
- **OpenAI Compatible**: Standard API format

## Performance

| Feature | vLLM | Standard |
|---------|------|----------|
| Throughput | 24x higher | Baseline |
| Memory | 90% efficient | 50-60% |
| Batch Size | Dynamic | Fixed |

## Supported Models

- Llama 2/3/3.1/3.2
- Mistral/Mixtral
- Qwen/Qwen2
- Falcon
- MPT
- Phi-2/3
- And many more...

## Example: High-Throughput Processing

```python
# vLLM excels at batch processing
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="vllm",
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        api_base="http://localhost:8000/v1"
    )
)

async with Sentimatrix(config) as sm:
    # Process large batches efficiently
    results = await sm.analyze_batch(
        reviews,  # 10,000+ reviews
        batch_size=256
    )
```

## Docker Deployment

```bash
docker run --gpus all \
    -p 8000:8000 \
    vllm/vllm-openai:latest \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct
```

## Related

- [Provider Overview](index.md)
- [Ollama](ollama.md)
- [LM Studio](lmstudio.md)
- [Selection Guide](selection-guide.md)
