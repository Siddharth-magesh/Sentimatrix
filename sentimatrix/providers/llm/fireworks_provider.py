"""
Fireworks AI LLM Provider

Implements the BaseLLMProvider interface for Fireworks AI.
Optimized inference with FireAttention technology.

Official API Documentation: https://docs.fireworks.ai/
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
                "httpx package is required for Fireworks provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default model
DEFAULT_MODEL = "accounts/fireworks/models/llama-v3p1-70b-instruct"

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Llama 3 Series
    "accounts/fireworks/models/llama-v3p3-70b-instruct": {
        "context": 128000, "vision": False,
        "description": "Llama 3.3 70B - Latest"
    },
    "accounts/fireworks/models/llama-v3p1-405b-instruct": {
        "context": 128000, "vision": False,
        "description": "Llama 3.1 405B - Largest open model"
    },
    "accounts/fireworks/models/llama-v3p1-70b-instruct": {
        "context": 128000, "vision": False,
        "description": "Llama 3.1 70B"
    },
    "accounts/fireworks/models/llama-v3p1-8b-instruct": {
        "context": 128000, "vision": False,
        "description": "Llama 3.1 8B"
    },
    "accounts/fireworks/models/llama-v3p2-11b-vision-instruct": {
        "context": 128000, "vision": True,
        "description": "Llama 3.2 11B Vision"
    },
    "accounts/fireworks/models/llama-v3p2-90b-vision-instruct": {
        "context": 128000, "vision": True,
        "description": "Llama 3.2 90B Vision"
    },
    # Mixtral
    "accounts/fireworks/models/mixtral-8x22b-instruct": {
        "context": 65536, "vision": False,
        "description": "Mixtral 8x22B"
    },
    "accounts/fireworks/models/mixtral-8x7b-instruct": {
        "context": 32768, "vision": False,
        "description": "Mixtral 8x7B"
    },
    # Qwen
    "accounts/fireworks/models/qwen2p5-72b-instruct": {
        "context": 32768, "vision": False,
        "description": "Qwen 2.5 72B"
    },
    # DeepSeek
    "accounts/fireworks/models/deepseek-v3": {
        "context": 64000, "vision": False,
        "description": "DeepSeek V3"
    },
    # Embeddings
    "nomic-ai/nomic-embed-text-v1.5": {
        "context": 8192, "vision": False,
        "description": "Nomic embeddings"
    },
}


class FireworksProvider(BaseLLMProvider):
    """
    Fireworks AI LLM Provider.

    Provides optimized inference with FireAttention:
    - Fast inference speeds
    - Large model support (405B)
    - Vision models
    - Function calling
    - JSON mode

    Supports:
    - Chat completions
    - Streaming responses
    - Function/tool calling
    - Vision (select models)
    - Text embeddings
    - JSON mode

    Example:
        >>> config = LLMConfig(
        ...     provider="fireworks",
        ...     api_key="your-api-key",
        ...     model="accounts/fireworks/models/llama-v3p1-70b-instruct",
        ... )
        >>> async with FireworksProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    API_BASE = "https://api.fireworks.ai/inference/v1"

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Fireworks provider.

        Args:
            config: LLM configuration with Fireworks API key.
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
            name="fireworks",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Fireworks AI - Optimized inference with FireAttention",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=True,
                vision=model_config.get("vision", False),
                json_mode=True,
                embeddings=True,
                max_context_tokens=context_limit,
                max_output_tokens=16384,
            ),
            supported_models=list(MODEL_CONFIGS.keys()),
            website="https://fireworks.ai",
            documentation="https://docs.fireworks.ai/",
        )

    async def initialize(self) -> None:
        """Initialize the Fireworks client."""
        if self._initialized:
            return

        api_key = self._api_key or os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "fireworks",
                "API key not provided. Set FIREWORKS_API_KEY environment variable "
                "or pass api_key in config."
            )

        httpx = _get_httpx()
        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout if self._config else 120,
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the Fireworks client."""
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
        Generate a completion using Fireworks AI.

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

        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]

        for key in ["top_p", "frequency_penalty", "presence_penalty"]:
            if key in kwargs:
                payload[key] = kwargs[key]

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
                provider="fireworks",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("fireworks", str(e)) from e
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
        """Stream a completion using Fireworks AI."""
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
            raise ProviderError("fireworks", str(e)) from e

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings using Fireworks AI.

        Args:
            texts: List of texts to embed
            model: Embedding model (default: nomic-embed)

        Returns:
            List of embedding vectors
        """
        self._ensure_initialized()

        embed_model = model or "nomic-ai/nomic-embed-text-v1.5"

        try:
            response = await self._client.post(
                "/embeddings",
                json={
                    "model": embed_model,
                    "input": texts,
                }
            )

            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            return [item["embedding"] for item in data["data"]]

        except Exception as e:
            raise ProviderError("fireworks", f"Embedding failed: {e}") from e

    def _handle_http_error(self, response) -> None:
        """Handle HTTP error responses."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 401:
            raise AuthenticationError("fireworks", error_msg)
        elif status == 429:
            raise RateLimitError("fireworks", retry_after=60)
        elif status == 400:
            if "token" in error_msg.lower():
                raise TokenLimitExceededError("fireworks", 0, 0)
            raise InvalidResponseError("fireworks", error_msg)
        elif status == 404:
            raise InvalidModelError(self._model, "fireworks")
        else:
            raise ProviderError("fireworks", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("fireworks", ProviderType.LLM, FireworksProvider)
