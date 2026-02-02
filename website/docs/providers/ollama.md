---
title: Ollama
description: Run LLMs locally with Ollama and Sentimatrix
---

# Ollama

Ollama allows you to run large language models locally on your machine. Perfect for privacy-sensitive applications and offline use.

<span class="status stable">Stable</span>

## Quick Facts

| Property | Value |
|----------|-------|
| **Cost** | Free (local hardware) |
| **Privacy** | Full (no data leaves your machine) |
| **Models** | LLaMA, Mistral, Phi, Gemma, and 100+ more |
| **Streaming** | :material-check: Supported |
| **Functions** | :material-check: Supported |
| **Vision** | :material-check: Supported (LLaVA, etc.) |
| **Embeddings** | :material-check: Supported |

## Setup

### Install Ollama

=== "macOS"

    ```bash
    brew install ollama
    ```

=== "Linux"

    ```bash
    curl -fsSL https://ollama.com/install.sh | sh
    ```

=== "Windows"

    Download from [ollama.com/download](https://ollama.com/download)

### Start the Server

```bash
ollama serve
```

The server runs on `http://localhost:11434` by default.

### Pull a Model

```bash
# LLaMA 3.2 (recommended for general use)
ollama pull llama3.2

# Smaller, faster model
ollama pull phi3

# Vision-capable model
ollama pull llava
```

### Configure Sentimatrix

=== "Environment Variable"

    ```bash
    export OLLAMA_HOST="http://localhost:11434"
    ```

=== "Python"

    ```python
    from sentimatrix.config import SentimatrixConfig, LLMConfig

    config = SentimatrixConfig(
        llm=LLMConfig(
            provider="ollama",
            base_url="http://localhost:11434",
            model="llama3.2"
        )
    )
    ```

=== "YAML"

    ```yaml
    llm:
      provider: ollama
      base_url: http://localhost:11434
      model: llama3.2
    ```

## Available Models

| Model | Size | RAM Required | Best For |
|-------|------|--------------|----------|
| `llama3.2` | 3B | 4GB | General use, fast |
| `llama3.2:1b` | 1B | 2GB | Ultra-fast, basic tasks |
| `llama3.1` | 8B | 8GB | Better quality |
| `llama3.1:70b` | 70B | 48GB+ | Best quality |
| `mistral` | 7B | 8GB | Code, reasoning |
| `phi3` | 3.8B | 4GB | Fast, efficient |
| `gemma2` | 9B | 12GB | Google's model |
| `llava` | 7B | 8GB | Vision tasks |

View all models: [ollama.com/library](https://ollama.com/library)

## Usage Examples

### Basic Usage

```python
import asyncio
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="ollama",
        model="llama3.2"
    )
)

async def main():
    async with Sentimatrix(config) as sm:
        summary = await sm.summarize_reviews(reviews)
        print(summary)

asyncio.run(main())
```

### Vision Analysis (LLaVA)

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="ollama",
        model="llava"
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze_image(
        image_path="product.jpg",
        prompt="What emotions does this product image convey?"
    )
```

### Custom Model Configuration

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="ollama",
        model="llama3.1",
        temperature=0.3,      # Lower for more focused output
        num_ctx=8192,         # Context window size
        num_gpu=1,            # GPU layers
    )
)
```

### Embeddings

```python
async with Sentimatrix(config) as sm:
    embeddings = await sm.get_embeddings([
        "Great product!",
        "Terrible experience.",
        "It's okay."
    ])
```

## Hardware Requirements

### Minimum Requirements

| Model Size | RAM | GPU VRAM | CPU |
|------------|-----|----------|-----|
| 1-3B | 4GB | Optional | 4 cores |
| 7-8B | 8GB | 6GB | 8 cores |
| 13B | 16GB | 10GB | 8 cores |
| 70B | 48GB+ | 40GB+ | 16 cores |

### GPU Acceleration

Ollama automatically uses GPU when available:

```bash
# Check GPU status
ollama list

# Force CPU only
OLLAMA_NUM_GPU=0 ollama serve
```

## Configuration Options

```python
LLMConfig(
    provider="ollama",
    base_url="http://localhost:11434",
    model="llama3.2",

    # Model settings
    temperature=0.7,
    num_ctx=4096,          # Context window
    num_predict=512,       # Max tokens to generate
    top_k=40,
    top_p=0.9,
    repeat_penalty=1.1,

    # Hardware
    num_gpu=-1,            # Auto-detect GPU layers
    num_thread=None,       # CPU threads (auto)

    # Reliability
    timeout=120,           # Longer timeout for large models
    max_retries=3,
)
```

## Model Management

### List Installed Models

```bash
ollama list

# NAME              SIZE      MODIFIED
# llama3.2          2.0 GB    2 hours ago
# mistral           4.1 GB    1 day ago
```

### Pull Models

```bash
# Latest version
ollama pull llama3.2

# Specific version
ollama pull llama3.2:latest

# Quantized version (smaller, faster)
ollama pull llama3.2:q4_0
```

### Remove Models

```bash
ollama rm mistral
```

### Create Custom Models

```bash
# Create Modelfile
cat << 'EOF' > Modelfile
FROM llama3.2

PARAMETER temperature 0.3
PARAMETER num_ctx 8192

SYSTEM You are a sentiment analysis expert. Always respond with structured analysis.
EOF

# Create model
ollama create sentiment-analyst -f Modelfile
```

## Remote Access

### Expose Ollama on Network

```bash
# Set environment variable
export OLLAMA_HOST="0.0.0.0:11434"
ollama serve
```

### Connect Remotely

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="ollama",
        base_url="http://192.168.1.100:11434",
        model="llama3.2"
    )
)
```

## Best Practices

1. **Choose the Right Model Size**
    - 1-3B for fast responses, basic tasks
    - 7-8B for balanced quality/speed
    - 70B+ for best quality (requires powerful hardware)

2. **Use GPU Acceleration**
    - Significantly faster than CPU
    - Check with `nvidia-smi` or `ollama list`

3. **Adjust Context Window**
    - Larger context = more memory
    - Match to your use case

4. **Use Quantized Models for Speed**
    ```bash
    ollama pull llama3.2:q4_0  # Fastest
    ollama pull llama3.2:q8_0  # Better quality
    ```

## Troubleshooting

??? question "Connection refused"

    Ensure Ollama is running:

    ```bash
    ollama serve
    ```

??? question "Model not found"

    Pull the model first:

    ```bash
    ollama pull llama3.2
    ```

??? question "Out of memory"

    - Use a smaller model
    - Use quantized version (`q4_0`)
    - Reduce context window

    ```python
    LLMConfig(model="llama3.2:q4_0", num_ctx=2048)
    ```

??? question "Slow responses"

    - Use GPU acceleration
    - Use smaller/quantized model
    - Reduce `num_predict`

## Related

- [Provider Overview](index.md)
- [LM Studio](lmstudio.md) - GUI alternative
- [vLLM](vllm.md) - Production server
