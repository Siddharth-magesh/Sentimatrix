---
title: Text Generation WebUI
description: Integration with oobabooga's text-generation-webui
---

# Text Generation WebUI

Text Generation WebUI (oobabooga) is a popular Gradio-based interface for running LLMs locally with extensive model support.

## Quick Start

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="textgen",
        model="loaded-model",
        api_base="http://localhost:5000/v1"
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

## Setup

### Install Text Generation WebUI

```bash
git clone https://github.com/oobabooga/text-generation-webui
cd text-generation-webui
pip install -r requirements.txt
```

### Start with API

```bash
python server.py --api --listen
```

## Configuration

```python
LLMConfig(
    provider="textgen",
    model="loaded-model",
    api_base="http://localhost:5000/v1",
    temperature=0.7,
    max_tokens=4096,
    timeout=120,
)
```

## Features

- **Multi-Format Support**: GGUF, GPTQ, AWQ, EXL2
- **Multiple Loaders**: Transformers, llama.cpp, ExLlamaV2, AutoGPTQ
- **Character/Chat Modes**: Various interaction styles
- **Extensions**: LoRA, multimodal, TTS
- **OpenAI API**: Compatible endpoint

## Supported Model Formats

| Format | Loader | Best For |
|--------|--------|----------|
| GGUF | llama.cpp | CPU/mixed |
| GPTQ | AutoGPTQ | GPU |
| AWQ | AutoAWQ | GPU |
| EXL2 | ExLlamaV2 | Fast GPU |
| FP16 | Transformers | Full precision |

## Example: Multi-Format Support

```python
# Text-gen-webui supports many formats
# Just load the model in the UI, then connect

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="textgen",
        api_base="http://localhost:5000/v1"
    )
)

async with Sentimatrix(config) as sm:
    result = await sm.analyze("Great product!")
```

## API Endpoints

The OpenAI-compatible API provides:

- `/v1/chat/completions` - Chat completions
- `/v1/completions` - Text completions
- `/v1/models` - List loaded models

## Launch Options

```bash
# With specific loader
python server.py --api --loader exllamav2

# With model auto-load
python server.py --api --model llama-3.1-8b

# With extensions
python server.py --api --extensions openai
```

## Related

- [Provider Overview](index.md)
- [Ollama](ollama.md)
- [ExLlamaV2](exllamav2.md)
- [Selection Guide](selection-guide.md)
