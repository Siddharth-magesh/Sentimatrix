"""
vLLM LLM Provider

Implements the BaseLLMProvider interface for vLLM.
High-throughput serving with PagedAttention for any HuggingFace model.

Official Documentation: https://docs.vllm.ai/
"""

from __future__ import annotations

import time
from typing import Any, AsyncIterator, Dict, List, Optional

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
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
                "httpx package is required for vLLM provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default settings
DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "local-model"


class VLLMProvider(BaseLLMProvider):
    """
    vLLM LLM Provider.

    Provides high-throughput local inference with vLLM:
    - PagedAttention for efficient memory management
    - Continuous batching for high throughput
    - OpenAI-compatible API server
    - Support for any HuggingFace model

    vLLM Setup:
    1. Install: pip install vllm
    2. Start server:
       python -m vllm.entrypoints.openai.api_server \\
           --model meta-llama/Llama-2-7b-chat-hf \\
           --port 8000
    3. Use this provider to connect

    Supports:
    - Chat completions
    - Streaming responses
    - System prompts
    - Batch inference (high throughput)
    - Speculative decoding

    Example:
        >>> config = LLMConfig(
        ...     provider="vllm",
        ...     model="meta-llama/Llama-2-7b-chat-hf",
        ...     base_url="http://localhost:8000/v1",
        ... )
        >>> async with VLLMProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize vLLM provider.

        Args:
            config: LLM configuration. base_url defaults to localhost:8000.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._base_url = getattr(config, 'base_url', None) or DEFAULT_BASE_URL

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="vllm",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="vLLM - High-throughput LLM serving with PagedAttention",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=True,
                vision=True,  # Model dependent
                json_mode=True,
                embeddings=False,  # vLLM focuses on generation
                max_context_tokens=128000,  # Model dependent
                max_output_tokens=4096,
            ),
            supported_models=["any HuggingFace model"],
            website="https://vllm.ai",
            documentation="https://docs.vllm.ai/",
        )

    async def initialize(self) -> None:
        """Initialize the vLLM client."""
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
        """Close the vLLM client."""
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
        Generate a completion using vLLM.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional vLLM-specific parameters

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

        # vLLM-specific parameters
        for key in [
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "best_of",
            "use_beam_search",
            "length_penalty",
            "early_stopping",
            "ignore_eos",
            "skip_special_tokens",
            "seed",
        ]:
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
                provider="vllm",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("vllm", str(e)) from e
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
        """Stream a completion using vLLM."""
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

        # vLLM-specific streaming parameters
        for key in ["top_p", "top_k", "repetition_penalty", "seed"]:
            if key in kwargs:
                payload[key] = kwargs[key]

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
            raise ProviderError("vllm", str(e)) from e

    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion using vLLM completions endpoint.

        This is a raw text completion (not chat) endpoint.

        Args:
            prompt: Text prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        payload = {
            "model": self._model,
            "prompt": prompt,
            "temperature": temp,
            "max_tokens": max_tok,
        }

        if stop:
            payload["stop"] = stop

        for key in ["top_p", "top_k", "frequency_penalty", "presence_penalty", "seed"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            response = await self._client.post(
                "/completions",
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
                content=choice["text"] or "",
                model=data.get("model", self._model),
                provider="vllm",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("vllm", str(e)) from e
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from vLLM server.

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
            raise ProviderError("vllm", f"Failed to list models: {e}") from e

    def _handle_http_error(self, response) -> None:
        """Handle HTTP error responses."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 404:
            raise InvalidModelError(self._model, "vllm")
        elif status == 400:
            raise InvalidResponseError("vllm", error_msg)
        else:
            raise ProviderError("vllm", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("vllm", ProviderType.LLM, VLLMProvider)
