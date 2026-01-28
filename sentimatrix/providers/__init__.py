"""
Sentimatrix Providers Module

Contains provider interfaces and implementations for:
- LLM providers (OpenAI, Anthropic, Groq, etc.)
- Scraper providers (Playwright, Selenium, API-based)
- Model providers (HuggingFace sentiment/emotion models)
"""

from sentimatrix.providers.base import (
    BaseLLMProvider,
    BaseModelProvider,
    BaseScraperProvider,
    ProviderCapabilities,
    ProviderInfo,
    ProviderRegistry,
    get_provider,
    register_provider,
)

__all__ = [
    "BaseLLMProvider",
    "BaseScraperProvider",
    "BaseModelProvider",
    "ProviderInfo",
    "ProviderCapabilities",
    "ProviderRegistry",
    "get_provider",
    "register_provider",
]
