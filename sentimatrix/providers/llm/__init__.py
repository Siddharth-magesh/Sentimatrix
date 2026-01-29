"""
Sentimatrix LLM Providers

Provider implementations for various LLM services:

Core Providers:
- OpenAI (GPT-4o, GPT-4o-mini, o1)
- Anthropic (Claude 3.5 Sonnet, Claude 3)
- Google Gemini (Gemini 2.0, 1.5 Pro/Flash)

Cloud Enterprise:
- Azure OpenAI (Enterprise OpenAI via Microsoft Azure)
- Amazon Bedrock (AWS multi-model: Claude, Llama, Titan, Mistral)

Fast Inference:
- Groq (LPU-accelerated: LLaMA, Mixtral)
- Cerebras (WSE: 1800 tok/s)
- Fireworks AI (FireAttention optimized)
- Together AI (200+ open source models)

Router/Gateway:
- OpenRouter (Unified gateway to 200+ models)

Specialized:
- Mistral (European AI, code, embeddings)
- Cohere (RAG, embeddings, reranking)
- DeepSeek (Advanced reasoning and coding)

Local Inference:
- Ollama (Local models, model management)
- LM Studio (GGUF models, GUI)
- vLLM (High-throughput serving, PagedAttention)
- llama.cpp (CPU/GPU optimized GGUF inference)
- text-generation-webui (Multi-backend, GUI)
- ExLlamaV2 (Ultra-fast GPTQ/EXL2 inference)

Usage:
    >>> from sentimatrix.providers.llm import OpenAIProvider, LLMProviderManager
    >>> from sentimatrix.core.config import LLMConfig

    # Single provider
    >>> config = LLMConfig(provider="openai", api_key="sk-...", model="gpt-4o-mini")
    >>> async with OpenAIProvider(config) as provider:
    ...     response = await provider.generate("Hello!")
    ...     print(response.content)

    # Multiple providers with fallback
    >>> manager = LLMProviderManager()
    >>> manager.add_provider("openai", LLMConfig(provider="openai", api_key="sk-..."))
    >>> manager.add_provider("groq", LLMConfig(provider="groq", api_key="gsk_..."))
    >>> async with manager:
    ...     response = await manager.generate("Hello!")
"""

# Core Providers
from sentimatrix.providers.llm.openai_provider import OpenAIProvider
from sentimatrix.providers.llm.anthropic_provider import AnthropicProvider
from sentimatrix.providers.llm.gemini_provider import GeminiProvider

# Cloud Enterprise
from sentimatrix.providers.llm.azure_openai_provider import AzureOpenAIProvider
from sentimatrix.providers.llm.bedrock_provider import BedrockProvider

# Fast Inference
from sentimatrix.providers.llm.groq_provider import GroqProvider
from sentimatrix.providers.llm.cerebras_provider import CerebrasProvider
from sentimatrix.providers.llm.fireworks_provider import FireworksProvider
from sentimatrix.providers.llm.together_provider import TogetherProvider

# Router/Gateway
from sentimatrix.providers.llm.openrouter_provider import OpenRouterProvider

# Specialized
from sentimatrix.providers.llm.mistral_provider import MistralProvider
from sentimatrix.providers.llm.cohere_provider import CohereProvider
from sentimatrix.providers.llm.deepseek_provider import DeepSeekProvider

# Local Inference
from sentimatrix.providers.llm.ollama_provider import OllamaProvider
from sentimatrix.providers.llm.lmstudio_provider import LMStudioProvider
from sentimatrix.providers.llm.vllm_provider import VLLMProvider
from sentimatrix.providers.llm.llamacpp_provider import LlamaCppProvider
from sentimatrix.providers.llm.textgen_provider import TextGenProvider
from sentimatrix.providers.llm.exllamav2_provider import ExLlamaV2Provider

# Manager
from sentimatrix.providers.llm.manager import (
    LLMProviderManager,
    LLMManagerConfig,
    ProviderConfig,
    ProviderHealth,
    FallbackStrategy,
    create_manager_from_config,
)

__all__ = [
    # Core Providers
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    # Cloud Enterprise
    "AzureOpenAIProvider",
    "BedrockProvider",
    # Fast Inference
    "GroqProvider",
    "CerebrasProvider",
    "FireworksProvider",
    "TogetherProvider",
    # Router/Gateway
    "OpenRouterProvider",
    # Specialized
    "MistralProvider",
    "CohereProvider",
    "DeepSeekProvider",
    # Local Inference
    "OllamaProvider",
    "LMStudioProvider",
    "VLLMProvider",
    "LlamaCppProvider",
    "TextGenProvider",
    "ExLlamaV2Provider",
    # Manager
    "LLMProviderManager",
    "LLMManagerConfig",
    "ProviderConfig",
    "ProviderHealth",
    "FallbackStrategy",
    "create_manager_from_config",
]
