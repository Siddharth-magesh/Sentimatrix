"""
Azure OpenAI LLM Provider

Implements the BaseLLMProvider interface for Azure OpenAI Service.
Provides enterprise-grade access to OpenAI models via Microsoft Azure.

Official API Documentation: https://learn.microsoft.com/en-us/azure/ai-services/openai/
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    AuthenticationError,
    ContentFilteredError,
    InvalidModelError,
    InvalidResponseError,
    RateLimitError,
    TokenLimitExceededError,
    ProviderError,
)
from sentimatrix.providers.base import (
    BaseLLMProvider,
    LLMResponse,
    ProviderCapabilities,
    ProviderInfo,
    ProviderType,
    TokenUsage,
    register_provider,
)

# Lazy imports
_openai = None


def _get_openai():
    """Lazy import of openai module."""
    global _openai
    if _openai is None:
        try:
            import openai
            _openai = openai
        except ImportError:
            raise ImportError(
                "openai package is required for Azure OpenAI provider. "
                "Install it with: pip install openai"
            )
    return _openai


# Default deployment configurations
DEFAULT_API_VERSION = "2024-02-01"

# Model context limits (deployment-dependent)
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-35-turbo": 16385,
    "gpt-35-turbo-16k": 16385,
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
}

# Feature support
VISION_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo"}
FUNCTION_CALLING_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-4-32k",
    "gpt-35-turbo", "gpt-35-turbo-16k"
}
JSON_MODE_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo"}


class AzureOpenAIProvider(BaseLLMProvider):
    """
    Azure OpenAI LLM Provider.

    Provides enterprise-grade access to OpenAI models through Azure.

    Supports:
    - Chat completions with Azure deployments
    - Streaming responses
    - Function/tool calling
    - JSON mode
    - Vision (image inputs)
    - Text embeddings
    - Enterprise security and compliance

    Example:
        >>> config = LLMConfig(
        ...     provider="azure-openai",
        ...     api_key="your-azure-key",
        ...     api_base="https://your-resource.openai.azure.com/",
        ...     model="gpt-4o",  # Your deployment name
        ... )
        >>> async with AzureOpenAIProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Azure OpenAI provider.

        Args:
            config: LLM configuration with Azure-specific settings:
                - api_key: Azure OpenAI API key
                - api_base: Azure endpoint URL
                - model: Deployment name
                - api_version: API version (optional)
        """
        super().__init__(config)
        self._client: Any = None
        self._async_client: Any = None
        self._deployment = config.model if config else "gpt-4o"
        self._api_key = config.api_key if config else None
        self._api_base = config.api_base if config else None
        self._api_version = getattr(config, 'api_version', None) or DEFAULT_API_VERSION

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        deployment = self._deployment
        context_limit = MODEL_CONTEXT_LIMITS.get(deployment, 128000)

        return ProviderInfo(
            name="azure-openai",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Azure OpenAI Service - Enterprise GPT models",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=any(m in deployment for m in FUNCTION_CALLING_MODELS),
                vision=any(m in deployment for m in VISION_MODELS),
                json_mode=any(m in deployment for m in JSON_MODE_MODELS),
                embeddings=True,
                max_context_tokens=context_limit,
                max_output_tokens=min(16384, context_limit),
            ),
            supported_models=list(MODEL_CONTEXT_LIMITS.keys()),
            website="https://azure.microsoft.com/products/ai-services/openai-service",
            documentation="https://learn.microsoft.com/azure/ai-services/openai/",
        )

    async def initialize(self) -> None:
        """Initialize the Azure OpenAI client."""
        if self._initialized:
            return

        openai = _get_openai()

        # Resolve credentials
        api_key = self._api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        api_base = self._api_base or os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_version = self._api_version or os.environ.get(
            "AZURE_OPENAI_API_VERSION", DEFAULT_API_VERSION
        )

        if not api_key:
            raise AuthenticationError(
                "azure-openai",
                "API key not provided. Set AZURE_OPENAI_API_KEY environment variable "
                "or pass api_key in config."
            )

        if not api_base:
            raise AuthenticationError(
                "azure-openai",
                "Azure endpoint not provided. Set AZURE_OPENAI_ENDPOINT environment variable "
                "or pass api_base in config."
            )

        # Create Azure clients
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "api_version": api_version,
            "azure_endpoint": api_base,
            "timeout": self._config.timeout if self._config else 60,
        }

        self._async_client = openai.AsyncAzureOpenAI(**client_kwargs)
        self._client = openai.AzureOpenAI(**client_kwargs)
        self._initialized = True

    async def close(self) -> None:
        """Close the Azure OpenAI client."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
        if self._client:
            self._client.close()
            self._client = None
        self._initialized = False

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion using Azure OpenAI.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Build messages
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self._deployment,
            "messages": messages,
        }

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        params["temperature"] = temp

        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )
        params["max_tokens"] = max_tok

        if stop:
            params["stop"] = stop

        # Handle response_format for JSON mode
        if "response_format" in kwargs:
            params["response_format"] = kwargs.pop("response_format")

        # Handle additional kwargs
        for key in ["top_p", "frequency_penalty", "presence_penalty", "seed"]:
            if key in kwargs:
                params[key] = kwargs.pop(key)

        try:
            response = await self._async_client.chat.completions.create(**params)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return LLMResponse(
                content=content,
                model=self._deployment,
                provider="azure-openai",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_error(e)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion using Azure OpenAI."""
        self._ensure_initialized()

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        params: Dict[str, Any] = {
            "model": self._deployment,
            "messages": messages,
            "stream": True,
        }

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        params["temperature"] = temp

        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )
        params["max_tokens"] = max_tok

        if stop:
            params["stop"] = stop

        try:
            stream = await self._async_client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._handle_error(e)

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings using Azure OpenAI.

        Args:
            texts: List of texts to embed
            model: Embedding model deployment name

        Returns:
            List of embedding vectors
        """
        self._ensure_initialized()

        embed_model = model or "text-embedding-ada-002"

        try:
            response = await self._async_client.embeddings.create(
                input=texts,
                model=embed_model,
            )

            return [item.embedding for item in response.data]

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """Handle Azure OpenAI API errors."""
        openai = _get_openai()

        error_msg = str(error)

        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError("azure-openai", error_msg) from error
        elif isinstance(error, openai.RateLimitError):
            raise RateLimitError("azure-openai", retry_after=60) from error
        elif isinstance(error, openai.BadRequestError):
            if "content_filter" in error_msg.lower():
                raise ContentFilteredError("azure-openai", error_msg) from error
            elif "context_length" in error_msg.lower() or "token" in error_msg.lower():
                raise TokenLimitExceededError("azure-openai", 0, 0) from error
            raise InvalidResponseError("azure-openai", error_msg) from error
        elif isinstance(error, openai.NotFoundError):
            raise InvalidModelError(self._deployment, "azure-openai") from error
        else:
            raise ProviderError("azure-openai", error_msg) from error


# Register the provider
register_provider("azure-openai", ProviderType.LLM, AzureOpenAIProvider)
