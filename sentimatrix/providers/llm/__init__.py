"""
Sentimatrix LLM Providers

Provider implementations for various LLM services:
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude 3.5 Sonnet, Claude 3)
- Google Gemini (Gemini 2.0, 1.5 Pro/Flash)
- Groq (LLaMA, Mixtral - fast inference)
- Ollama (local models)

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

from sentimatrix.providers.llm.openai_provider import OpenAIProvider
from sentimatrix.providers.llm.groq_provider import GroqProvider
from sentimatrix.providers.llm.anthropic_provider import AnthropicProvider
from sentimatrix.providers.llm.ollama_provider import OllamaProvider
from sentimatrix.providers.llm.gemini_provider import GeminiProvider
from sentimatrix.providers.llm.manager import (
    LLMProviderManager,
    LLMManagerConfig,
    ProviderConfig,
    ProviderHealth,
    FallbackStrategy,
    create_manager_from_config,
)

__all__ = [
    # Providers
    "OpenAIProvider",
    "GroqProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "GeminiProvider",
    # Manager
    "LLMProviderManager",
    "LLMManagerConfig",
    "ProviderConfig",
    "ProviderHealth",
    "FallbackStrategy",
    "create_manager_from_config",
]
