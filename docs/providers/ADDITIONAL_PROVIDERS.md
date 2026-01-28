# Sentimatrix V2 - Additional LLM Providers

## Overview

This document lists additional LLM providers identified for potential integration. These are categorized by type and priority.

---

## Router/Gateway Services

These services route requests to multiple underlying providers.

| Provider | Description | API Type | Priority |
|----------|-------------|----------|----------|
| OpenRouter | Multi-provider router, unified API | REST | P1 |
| Vercel AI Gateway | Edge AI routing | REST | P2 |
| Cloudflare Workers AI | Edge inference | REST | P2 |
| Cloudflare AI Gateway | AI proxy/gateway | REST | P2 |
| FastRouter | Fast routing service | REST | P3 |
| ZenMux | Request multiplexer | REST | P3 |
| Helicone | LLM observability + proxy | REST | P3 |

**Integration Note:** Router services often use OpenAI-compatible APIs, simplifying integration.

---

## Chinese AI Providers

| Provider | Models | Description | Priority |
|----------|--------|-------------|----------|
| DeepSeek | V3, R1 | Cost-effective, reasoning | P1 |
| Alibaba Qwen | Qwen series | Large scale | P2 |
| Moonshot AI | Kimi | Long context | P2 |
| Zhipu AI | GLM series | ChatGLM | P2 |
| SiliconFlow | Various | Chinese infra | P2 |
| Baidu ERNIE | ERNIE series | Search integration | P3 |
| MiniMax | MiniMax models | Chinese LLM | P3 |
| Xiaomi | MiLM | Device AI | P3 |
| Bailing | Bailing models | Chinese | P3 |
| 302.AI | Various | Platform | P3 |
| Kimi For Coding | Code models | Coding focus | P3 |

---

## Enterprise/Cloud Providers

| Provider | Description | API Type | Priority |
|----------|-------------|----------|----------|
| Azure OpenAI | Microsoft OpenAI | REST | P1 |
| Azure Cognitive Services | Azure AI services | REST | P2 |
| Amazon Bedrock | AWS multi-model | REST | P1 |
| Google Vertex AI | GCP AI platform | REST | P2 |
| SAP AI Core | SAP enterprise | REST | P3 |
| OVHcloud AI | EU cloud | REST | P3 |
| Scaleway | EU cloud | REST | P3 |
| Vultr | Cloud GPU | REST | P3 |

---

## Open Source Hosting Platforms

| Provider | Models Available | Description | Priority |
|----------|------------------|-------------|----------|
| Together AI | 200+ | Wide selection | P1 |
| Hugging Face | 400K+ | Model hub | P1 |
| Replicate | Various | Per-second billing | P2 |
| Deep Infra | OSS models | Inference | P2 |
| Baseten | Custom models | Deployment | P2 |
| Modal | Serverless | GPU functions | P3 |
| IO.NET | Decentralized | Distributed compute | P3 |
| ModelScope | Alibaba models | Chinese hub | P3 |

---

## Specialized Inference

| Provider | Specialty | Speed | Priority |
|----------|-----------|-------|----------|
| Groq | LPU hardware | 750 tok/s | P0 |
| Cerebras | Wafer-scale | 1800 tok/s | P1 |
| SambaNova | RDU | 580 tok/s | P2 |
| Fireworks AI | FireAttention | Fast | P1 |
| Friendli | Optimization | Fast | P3 |

---

## Local Inference Solutions

| Provider | Type | Description | Priority |
|----------|------|-------------|----------|
| Ollama | Server | Easy local LLM | P0 |
| Ollama Cloud | Hosted | Cloud Ollama | P2 |
| LM Studio | Desktop | GUI application | P1 |
| vLLM | Server | Production server | P1 |
| llama.cpp | Library | CPU inference | P2 |
| Llama | Meta | Official models | P2 |
| NanoGPT | Minimal | Small models | P3 |

---

## Developer Tools & Platforms

| Provider | Type | Description | Priority |
|----------|------|-------------|----------|
| GitHub Copilot | Code | Code completion | P3 |
| GitHub Models | Hub | Model marketplace | P2 |
| GitLab Duo | Code | GitLab AI | P3 |
| Weights & Biases | MLOps | Experiment tracking | P3 |
| Poe | Consumer | Quora AI | P3 |
| v0 | Vercel | UI generation | P3 |

---

## Other Notable Providers

| Provider | Category | Description | Priority |
|----------|----------|-------------|----------|
| Cohere | Enterprise | RAG, embeddings | P1 |
| Mistral | EU | French AI | P0 |
| xAI (Grok) | Consumer | Elon's AI | P2 |
| Perplexity | Search | Search + LLM | P2 |
| Upstage | Korean | Solar models | P2 |
| Nvidia NIM | Infra | GPU optimized | P2 |
| NovitaAI | Platform | Various | P3 |
| Venice AI | Privacy | Privacy focused | P3 |
| Vivgrid | Platform | Various | P3 |
| Chutes | Platform | Various | P3 |
| Cortecs | Platform | Various | P3 |
| Moark | Platform | Various | P3 |
| Inception | Platform | Various | P3 |
| Privatemode AI | Privacy | Privacy focused | P3 |
| LucidQuery AI | Query | Query AI | P3 |
| Firmware | Platform | Various | P3 |
| Abacus | Platform | Various | P3 |
| Nebius | Token | Token Factory | P3 |
| iFlow | Workflow | Flow AI | P3 |
| Synthetic | Platform | Various | P3 |
| submodel | Platform | Various | P3 |
| Z.AI | Platform | Various | P3 |
| Inference | Platform | Various | P3 |
| Requesty | Platform | Various | P3 |
| Morph | Platform | Various | P3 |
| AIHubMix | Hub | Various | P3 |

---

## Integration Priority Summary

| Priority | Count | Focus |
|----------|-------|-------|
| P0 | 6 | Core providers (OpenAI, Anthropic, Gemini, Groq, Ollama, Mistral) |
| P1 | 11 | Important alternatives and features |
| P2 | 20 | Extended coverage |
| P3 | 48+ | Future/as-needed |

---

## API Compatibility Notes

Many providers use OpenAI-compatible APIs:
- Groq
- Cerebras
- Together AI
- Fireworks AI
- DeepSeek
- OpenRouter
- vLLM
- Ollama

This allows reusing the OpenAI provider implementation with different base URLs.

---

## Implementation Strategy

1. **Implement OpenAI provider first** - Reference implementation
2. **Create OpenAI-compatible base class** - Reuse for compatible providers
3. **Implement unique providers** - Anthropic, Google have different APIs
4. **Add router support** - OpenRouter gives access to many models
5. **Local providers** - Ollama, vLLM for offline use
