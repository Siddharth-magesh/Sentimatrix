"""
Cerebras LLM Provider

Implements the BaseLLMProvider interface for Cerebras Inference.
Ultra-fast inference using Wafer-Scale Engine (WSE) technology.

Official API Documentation: https://inference-docs.cerebras.ai/
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    AuthenticationError,
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
_httpx = None


def _get_httpx():
    """Lazy import of httpx module."""
    global _httpx
    if _httpx is None:
        try:
            import httpx
            _httpx = httpx
        except ImportError:
            raise ImportError(
                "httpx package is required for Cerebras provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default model
DEFAULT_MODEL = "llama3.1-70b"

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "llama3.1-8b": {
        "context": 128000,
        "speed": "1800 tok/s",
        "description": "Llama 3.1 8B - Fastest inference"
    },
    "llama3.1-70b": {
        "context": 128000,
        "speed": "450 tok/s",
        "description": "Llama 3.1 70B - Balanced performance"
    },
    "llama-3.3-70b": {
        "context": 128000,
        "speed": "450 tok/s",
        "description": "Llama 3.3 70B - Latest model"
    },
}


class CerebrasProvider(BaseLLMProvider):
    """
    Cerebras Inference LLM Provider.

    Provides ultra-fast inference using Wafer-Scale Engine (WSE):
    - 1800+ tokens/second for 8B models
    - 450+ tokens/second for 70B models
    - Instant response times
    - OpenAI-compatible API

    Supports:
    - Chat completions
    - Streaming responses
    - OpenAI-compatible interface

    Example:
        >>> config = LLMConfig(
        ...     provider="cerebras",
        ...     api_key="your-api-key",
        ...     model="llama3.1-70b",
        ... )
        >>> async with CerebrasProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    API_BASE = "https://api.cerebras.ai/v1"

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Cerebras provider.

        Args:
            config: LLM configuration with Cerebras API key.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._api_key = config.api_key if config else None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model_config = MODEL_CONFIGS.get(self._model, {})
        context_limit = model_config.get("context", 128000)

        return ProviderInfo(
            name="cerebras",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description=f"Cerebras WSE - Ultra-fast inference ({model_config.get('speed', '1000+ tok/s')})",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=False,
                vision=False,
                json_mode=True,
                embeddings=False,
                max_context_tokens=context_limit,
                max_output_tokens=8192,
            ),
            supported_models=list(MODEL_CONFIGS.keys()),
            website="https://cerebras.ai",
            documentation="https://inference-docs.cerebras.ai/",
        )

    async def initialize(self) -> None:
        """Initialize the Cerebras client."""
        if self._initialized:
            return

        api_key = self._api_key or os.environ.get("CEREBRAS_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "cerebras",
                "API key not provided. Set CEREBRAS_API_KEY environment variable "
                "or pass api_key in config."
            )

        httpx = _get_httpx()
        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout if self._config else 60,
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the Cerebras client."""
        if self._client:
            await self._client.aclose()
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
        Generate a completion using Cerebras.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature (0-1.5)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }

        if stop:
            payload["stop"] = stop

        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        try:
            response = await self._client.post(
                "/chat/completions",
                json=payload,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            choice = data["choices"][0]
            usage_data = data.get("usage", {})

            usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            )

            return LLMResponse(
                content=choice["message"]["content"] or "",
                model=data.get("model", self._model),
                provider="cerebras",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("cerebras", str(e)) from e
            raise

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion using Cerebras."""
        self._ensure_initialized()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": True,
        }

        if stop:
            payload["stop"] = stop

        try:
            async with self._client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            import json
                            data = json.loads(data_str)
                            if data.get("choices"):
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except Exception:
                            continue

        except Exception as e:
            raise ProviderError("cerebras", str(e)) from e

    def _handle_http_error(self, response) -> None:
        """Handle HTTP error responses."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 401:
            raise AuthenticationError("cerebras", error_msg)
        elif status == 429:
            raise RateLimitError("cerebras", retry_after=60)
        elif status == 400:
            if "token" in error_msg.lower():
                raise TokenLimitExceededError("cerebras", 0, 0)
            raise InvalidResponseError("cerebras", error_msg)
        elif status == 404:
            raise InvalidModelError(self._model, "cerebras")
        else:
            raise ProviderError("cerebras", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("cerebras", ProviderType.LLM, CerebrasProvider)
