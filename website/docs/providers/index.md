---
title: LLM Providers
description: Configure and use 19 different LLM providers with Sentimatrix
---

# LLM Providers

Sentimatrix supports **19 LLM providers** for enhanced sentiment analysis, summarization, and insight generation.

## Provider Categories

<div class="grid">

<div class="card">
<h3>:material-cloud: Cloud Providers</h3>
<p>Managed API services with enterprise reliability.</p>
<p>OpenAI, Anthropic, Google, Mistral, Cohere, Groq</p>
</div>

<div class="card">
<h3>:material-server: Inference Providers</h3>
<p>Cost-effective hosting for open-source models.</p>
<p>Together, Fireworks, OpenRouter, Cerebras, DeepSeek</p>
</div>

<div class="card">
<h3>:material-desktop-tower: Local Providers</h3>
<p>Run models locally for privacy and cost savings.</p>
<p>Ollama, LM Studio, vLLM, llama.cpp, ExLlamaV2</p>
</div>

<div class="card">
<h3>:material-office-building: Enterprise</h3>
<p>Enterprise-grade deployments with compliance.</p>
<p>Azure OpenAI, AWS Bedrock</p>
</div>

</div>

## Quick Comparison

| Provider | Best For | Pricing | Streaming | Vision |
|----------|----------|---------|-----------|--------|
| **OpenAI** | Quality & features | $$$ | :material-check: | :material-check: |
| **Anthropic** | Safety & reasoning | $$$ | :material-check: | :material-check: |
| **Google** | Multimodal & long context | $$ | :material-check: | :material-check: |
| **Groq** | Speed (free tier) | Free-$ | :material-check: | :material-close: |
| **Together** | OSS models | $ | :material-check: | :material-check: |
| **DeepSeek** | Value | $ | :material-check: | :material-close: |
| **Ollama** | Local/privacy | Free | :material-check: | :material-check: |

## Quick Start

### Cloud Provider (Groq - Free)

```python
from sentimatrix import Sentimatrix
from sentimatrix.config import SentimatrixConfig, LLMConfig

config = SentimatrixConfig(
    llm=LLMConfig(
        provider="groq",
        api_key="gsk_...",  # or use GROQ_API_KEY env var
        model="llama-3.3-70b-versatile"
    )
)

async with Sentimatrix(config) as sm:
    summary = await sm.summarize_reviews(reviews)
```

### Local Provider (Ollama)

```python
config = SentimatrixConfig(
    llm=LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434",
        model="llama3.2"
    )
)
```

## Feature Matrix

| Provider | Streaming | Functions | JSON Mode | Embeddings | Vision |
|----------|:---------:|:---------:|:---------:|:----------:|:------:|
| OpenAI | :material-check: | :material-check: | :material-check: | :material-check: | :material-check: |
| Anthropic | :material-check: | :material-check: | :material-check: | :material-close: | :material-check: |
| Google | :material-check: | :material-check: | :material-check: | :material-check: | :material-check: |
| Groq | :material-check: | :material-check: | :material-check: | :material-close: | :material-close: |
| Mistral | :material-check: | :material-check: | :material-check: | :material-check: | :material-close: |
| Cohere | :material-check: | :material-check: | :material-close: | :material-check: | :material-close: |
| Together | :material-check: | :material-check: | :material-check: | :material-check: | :material-check: |
| Fireworks | :material-check: | :material-check: | :material-check: | :material-check: | :material-check: |
| OpenRouter | :material-check: | :material-check: | :material-check: | :material-close: | :material-check: |
| Cerebras | :material-check: | :material-check: | :material-close: | :material-close: | :material-close: |
| DeepSeek | :material-check: | :material-check: | :material-check: | :material-check: | :material-close: |
| Ollama | :material-check: | :material-check: | :material-check: | :material-check: | :material-check: |
| LM Studio | :material-check: | :material-close: | :material-check: | :material-check: | :material-close: |
| vLLM | :material-check: | :material-close: | :material-check: | :material-close: | :material-close: |
| llama.cpp | :material-check: | :material-close: | :material-check: | :material-close: | :material-close: |
| ExLlamaV2 | :material-check: | :material-close: | :material-close: | :material-close: | :material-close: |
| Azure OpenAI | :material-check: | :material-check: | :material-check: | :material-check: | :material-check: |
| AWS Bedrock | :material-check: | :material-check: | :material-check: | :material-close: | :material-check: |

## Pricing Comparison

Approximate costs per 1M tokens:

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| **Groq** | LLaMA 3.3 70B | Free | Free |
| **Google** | Gemini 1.5 Flash | Free | Free |
| **DeepSeek** | DeepSeek V3 | $0.07 | $0.27 |
| **Together** | LLaMA 3.1 70B | $0.88 | $0.88 |
| **Fireworks** | LLaMA 3.1 70B | $0.90 | $0.90 |
| **OpenAI** | GPT-4o-mini | $0.15 | $0.60 |
| **Mistral** | Mistral Large | $2.00 | $6.00 |
| **OpenAI** | GPT-4o | $2.50 | $10.00 |
| **Anthropic** | Claude 3.5 Sonnet | $3.00 | $15.00 |
| **Anthropic** | Claude 3 Opus | $15.00 | $75.00 |

## Selection Guide

### By Use Case

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Getting Started** | Groq | Free tier, fast, good quality |
| **Production** | OpenAI / Anthropic | Reliability, support |
| **Cost Sensitive** | DeepSeek, Together | Low cost, good quality |
| **Privacy Required** | Ollama, vLLM | Local execution |
| **Enterprise** | Azure OpenAI, Bedrock | Compliance, SLAs |
| **Best Quality** | Claude 3.5 Sonnet, GPT-4o | State-of-the-art |
| **Speed Critical** | Groq, Cerebras | Ultra-low latency |

### By Budget

=== "Free / Minimal"

    - **Groq** - Free tier with generous limits
    - **Google Gemini** - Free tier available
    - **Ollama** - Free (local hardware required)
    - **DeepSeek** - Very low cost

=== "Low Budget ($10-50/mo)"

    - **Together AI** - $0.88/1M tokens
    - **Fireworks** - $0.90/1M tokens
    - **OpenAI GPT-4o-mini** - $0.15-0.60/1M tokens

=== "Production Budget"

    - **OpenAI GPT-4o** - Best balance of quality/cost
    - **Anthropic Claude 3.5 Sonnet** - Best for reasoning
    - **Google Gemini 1.5 Pro** - Best for long context

## Configuration

### Environment Variables

```bash
# Cloud providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
export GROQ_API_KEY="gsk_..."
export MISTRAL_API_KEY="..."
export COHERE_API_KEY="..."

# Inference providers
export TOGETHER_API_KEY="..."
export FIREWORKS_API_KEY="..."
export OPENROUTER_API_KEY="..."
export DEEPSEEK_API_KEY="..."

# Local providers
export OLLAMA_HOST="http://localhost:11434"
export LMSTUDIO_HOST="http://localhost:1234"
```

### YAML Configuration

```yaml title="sentimatrix.yaml"
llm:
  provider: groq
  model: llama-3.3-70b-versatile

  # Optional settings
  temperature: 0.7
  max_tokens: 4096
  timeout: 30

  # Fallback providers
  fallback:
    - provider: together
      model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
    - provider: ollama
      model: llama3.2
```

## Provider Documentation

Detailed documentation for each provider:

### Cloud Providers

- [OpenAI](openai.md) - GPT-4o, GPT-4o-mini, o1
- [Anthropic](anthropic.md) - Claude 3.5 Sonnet, Claude 3
- [Google Gemini](gemini.md) - Gemini 2.0 Flash, 1.5 Pro
- [Mistral](mistral.md) - Mistral 7B, 8x7B, Large
- [Cohere](cohere.md) - Command R, Command R+
- [Groq](groq.md) - LLaMA 3.3, Mixtral

### Inference Providers

- [Together AI](together.md) - 200+ models
- [Fireworks](fireworks.md) - Fast OSS inference
- [OpenRouter](openrouter.md) - Model aggregator
- [Cerebras](cerebras.md) - Ultra-fast inference
- [DeepSeek](deepseek.md) - DeepSeek V3, R1

### Local Providers

- [Ollama](ollama.md) - Local LLM server
- [LM Studio](lmstudio.md) - Desktop GUI
- [vLLM](vllm.md) - Production server
- [llama.cpp](llamacpp.md) - Portable inference
- [ExLlamaV2](exllamav2.md) - Quantized models

### Enterprise

- [Azure OpenAI](azure.md) - Microsoft Azure
- [AWS Bedrock](bedrock.md) - Amazon Web Services
