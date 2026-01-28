# Sentimatrix V2 - Local LLM Providers

## Overview

Local LLM inference provides privacy, cost savings, and offline capability. V2 supports multiple local inference solutions.

---

## 1. Ollama

**Website:** https://ollama.ai

**Best For:** Easy setup, development, desktop use

**Features:**
- Simple CLI and API
- Model library with one-command download
- Multi-model support
- GPU acceleration
- OpenAI-compatible API

**Supported Models:**
| Model | Size | VRAM Required |
|-------|------|---------------|
| llama3.1:8b | 4.7GB | 8GB |
| llama3.1:70b | 40GB | 48GB+ |
| mistral:7b | 4.1GB | 8GB |
| mixtral:8x7b | 26GB | 32GB |
| qwen2.5:7b | 4.4GB | 8GB |
| phi3:mini | 2.2GB | 4GB |
| llava:7b | 4.5GB | 8GB (vision) |

**Installation:**
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model
ollama pull llama3.1:8b

# Run
ollama run llama3.1:8b
```

**Implementation:**
```python
# Module: providers/llm/ollama_provider.py
import httpx

class OllamaProvider(BaseLLMProvider):
    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def generate(self, prompt: str, **kwargs) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7)
                    }
                },
                timeout=120
            )
            return response.json()["response"]

    async def generate_stream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": True},
                timeout=120
            ) as response:
                async for line in response.aiter_lines():
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]

    async def embed(self, text: str) -> List[float]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text}
            )
            return response.json()["embedding"]
```

**Configuration:**
```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "llama3.1:8b"
  timeout: 120
  options:
    temperature: 0.7
    num_ctx: 4096
    num_gpu: 99  # Use all available GPU layers
```

---

## 2. vLLM

**Repository:** https://github.com/vllm-project/vllm

**Best For:** Production deployment, high throughput

**Features:**
- PagedAttention (efficient memory)
- Continuous batching
- OpenAI-compatible API server
- Tensor parallelism
- Quantization support

**Performance:** ~793 tokens/second (Llama 70B on 4x A100)

**Installation:**
```bash
pip install vllm

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --port 8000
```

**Implementation:**
```python
# Module: providers/llm/vllm_provider.py
from openai import AsyncOpenAI

class VLLMProvider(BaseLLMProvider):
    def __init__(self, model: str, base_url: str = "http://localhost:8000/v1"):
        self.client = AsyncOpenAI(
            api_key="EMPTY",  # vLLM doesn't require API key
            base_url=base_url
        )
        self.model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7)
        )
        return response.choices[0].message.content
```

**Configuration:**
```yaml
vllm:
  base_url: "http://localhost:8000/v1"
  model: "meta-llama/Llama-3.1-8B-Instruct"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.9
```

---

## 3. llama.cpp

**Repository:** https://github.com/ggerganov/llama.cpp

**Best For:** CPU inference, edge devices, maximum portability

**Features:**
- Pure C/C++ implementation
- No dependencies
- CPU-optimized (AVX, AVX2, AVX512)
- Quantization (Q4, Q5, Q8)
- Runs on minimal hardware

**Binary Size:** ~90MB

**Installation:**
```bash
# Build from source
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Or use Python bindings
pip install llama-cpp-python
```

**Implementation:**
```python
# Module: providers/llm/llamacpp_provider.py
from llama_cpp import Llama

class LlamaCppProvider(BaseLLMProvider):
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = 0):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers
        )

    async def generate(self, prompt: str, **kwargs) -> str:
        output = self.llm(
            prompt,
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.7),
            stop=kwargs.get("stop", [])
        )
        return output["choices"][0]["text"]
```

**Configuration:**
```yaml
llamacpp:
  model_path: "/models/llama-3.1-8b-q4_k_m.gguf"
  n_ctx: 4096
  n_gpu_layers: 35  # Offload layers to GPU
  n_threads: 8
```

---

## 4. LM Studio

**Website:** https://lmstudio.ai

**Best For:** Desktop users, GUI preference

**Features:**
- Beautiful GUI
- Model discovery and download
- OpenAI-compatible API server
- Chat interface
- Model comparison

**API Server:**
LM Studio can expose an OpenAI-compatible API at `http://localhost:1234/v1`

**Implementation:**
```python
# Module: providers/llm/lmstudio_provider.py
# Uses same implementation as vLLM (OpenAI-compatible)

class LMStudioProvider(BaseLLMProvider):
    def __init__(self, model: str = "local-model"):
        self.client = AsyncOpenAI(
            api_key="lm-studio",
            base_url="http://localhost:1234/v1"
        )
        self.model = model
```

---

## 5. Text Generation Inference (TGI)

**Repository:** https://github.com/huggingface/text-generation-inference

**Best For:** Hugging Face ecosystem, production

**Features:**
- Hugging Face models
- Continuous batching
- Flash Attention
- Tensor parallelism
- Quantization

**Installation:**
```bash
# Docker
docker run --gpus all -p 8080:80 \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-3.1-8B-Instruct
```

---

## Hardware Requirements

### Minimum Requirements by Model Size

| Model Size | RAM | VRAM (GPU) | Storage |
|------------|-----|------------|---------|
| 7B (Q4) | 8GB | 6GB | 4GB |
| 7B (FP16) | 16GB | 14GB | 14GB |
| 13B (Q4) | 16GB | 10GB | 8GB |
| 70B (Q4) | 48GB | 40GB | 40GB |
| 70B (FP16) | 140GB | 140GB | 140GB |

### Recommended GPUs

| GPU | VRAM | Max Model (Q4) | Max Model (FP16) |
|-----|------|----------------|------------------|
| RTX 3060 | 12GB | 13B | 7B |
| RTX 3090 | 24GB | 30B | 13B |
| RTX 4090 | 24GB | 30B | 13B |
| A100 40GB | 40GB | 70B | 30B |
| A100 80GB | 80GB | 70B | 70B |

---

## Quantization Guide

| Format | Bits | Quality Loss | Speed | Size Reduction |
|--------|------|--------------|-------|----------------|
| FP16 | 16 | None | Baseline | 1x |
| Q8 | 8 | Minimal | +20% | 2x |
| Q5_K_M | 5 | Low | +40% | 3x |
| Q4_K_M | 4 | Moderate | +60% | 4x |
| Q3_K_M | 3 | Noticeable | +80% | 5x |
| Q2_K | 2 | High | +100% | 8x |

**Recommendation:** Q4_K_M offers best balance of quality and size

---

## Configuration Comparison

```yaml
local:
  # Choose one provider
  provider: "ollama"  # ollama, vllm, llamacpp, lmstudio

  ollama:
    base_url: "http://localhost:11434"
    model: "llama3.1:8b"

  vllm:
    base_url: "http://localhost:8000/v1"
    model: "meta-llama/Llama-3.1-8B-Instruct"

  llamacpp:
    model_path: "/models/llama-3.1-8b.gguf"
    n_gpu_layers: 35

  # Common settings
  timeout: 120
  max_tokens: 2048
```

---

## Selection Guide

| Scenario | Provider |
|----------|----------|
| Quick setup | Ollama |
| Production server | vLLM |
| CPU-only/Edge | llama.cpp |
| Desktop GUI | LM Studio |
| HuggingFace models | TGI |

---

## Performance Optimization

### GPU Optimization
1. Maximize GPU layer offloading
2. Use appropriate quantization
3. Enable flash attention
4. Use continuous batching

### CPU Optimization
1. Enable AVX2/AVX512
2. Use Q4 quantization
3. Adjust thread count
4. Use memory mapping

### Memory Optimization
1. Use quantized models
2. Set appropriate context length
3. Enable KV cache optimization
4. Use streaming for long outputs
