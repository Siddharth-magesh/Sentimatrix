"""
LM Studio LLM Provider

Implements the BaseLLMProvider interface for LM Studio.
Local inference using any GGUF model through LM Studio's OpenAI-compatible API.

Official Documentation: https://lmstudio.ai/docs
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    AuthenticationError,
    InvalidModelError,
    InvalidResponseError,
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
                "httpx package is required for LM Studio provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default settings
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "local-model"


class LMStudioProvider(BaseLLMProvider):
    """
    LM Studio LLM Provider.

    Provides local inference through LM Studio's OpenAI-compatible API:
    - Run any GGUF model locally
    - No API key required
    - Full privacy - data never leaves your machine
    - OpenAI-compatible interface

    LM Studio Setup:
    1. Download LM Studio from https://lmstudio.ai
    2. Download a model from the in-app model browser
    3. Start the local server (default: http://localhost:1234)
    4. Use this provider to connect

    Supports:
    - Chat completions
    - Streaming responses
    - System prompts
    - Temperature and sampling control

    Example:
        >>> config = LLMConfig(
        ...     provider="lmstudio",
        ...     model="local-model",  # or specific model name
        ...     base_url="http://localhost:1234/v1",
        ... )
        >>> async with LMStudioProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize LM Studio provider.

        Args:
            config: LLM configuration. base_url defaults to localhost:1234.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._base_url = getattr(config, 'base_url', None) or DEFAULT_BASE_URL

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="lmstudio",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="LM Studio - Local GGUF model inference",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=True,
                vision=True,  # Model dependent
                json_mode=True,
                embeddings=True,
                max_context_tokens=128000,  # Model dependent
                max_output_tokens=4096,
            ),
            supported_models=["local-model", "any GGUF model"],
            website="https://lmstudio.ai",
            documentation="https://lmstudio.ai/docs",
        )

    async def initialize(self) -> None:
        """Initialize the LM Studio client."""
        if self._initialized:
            return

        httpx = _get_httpx()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout if self._config else 300,  # Longer timeout for local
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the LM Studio client."""
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
        Generate a completion using LM Studio.

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

        for key in ["top_p", "frequency_penalty", "presence_penalty", "seed"]:
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
                provider="lmstudio",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("lmstudio", str(e)) from e
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
        """Stream a completion using LM Studio."""
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
            raise ProviderError("lmstudio", str(e)) from e

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings using LM Studio.

        Args:
            texts: List of texts to embed
            model: Embedding model (optional)

        Returns:
            List of embedding vectors
        """
        self._ensure_initialized()

        embed_model = model or self._model

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
            raise ProviderError("lmstudio", f"Embedding failed: {e}") from e

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from LM Studio.

        Returns:
            List of model information dictionaries
        """
        self._ensure_initialized()

        try:
            response = await self._client.get("/models")

            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            return data.get("data", [])

        except Exception as e:
            raise ProviderError("lmstudio", f"Failed to list models: {e}") from e

    def _handle_http_error(self, response) -> None:
        """Handle HTTP error responses."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 404:
            raise InvalidModelError(self._model, "lmstudio")
        elif status == 400:
            raise InvalidResponseError("lmstudio", error_msg)
        else:
            raise ProviderError("lmstudio", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("lmstudio", ProviderType.LLM, LMStudioProvider)
