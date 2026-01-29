"""
llama.cpp LLM Provider

Implements the BaseLLMProvider interface for llama.cpp server.
CPU/GPU optimized inference for GGUF models with llama.cpp's server mode.

Official Documentation: https://github.com/ggerganov/llama.cpp/tree/master/examples/server
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
                "httpx package is required for llama.cpp provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default settings
DEFAULT_BASE_URL = "http://localhost:8080"
DEFAULT_MODEL = "local-model"


class LlamaCppProvider(BaseLLMProvider):
    """
    llama.cpp Server LLM Provider.

    Provides local inference through llama.cpp's HTTP server:
    - Optimized GGUF model loading
    - CPU inference with SIMD optimizations
    - GPU acceleration (CUDA, Metal, Vulkan)
    - Efficient memory management
    - Grammar-based structured outputs

    llama.cpp Server Setup:
    1. Build llama.cpp: cmake -B build && cmake --build build --config Release
    2. Start server:
       ./build/bin/server -m model.gguf --host 0.0.0.0 --port 8080
    3. Or with GPU: ./build/bin/server -m model.gguf -ngl 99 --port 8080
    4. Use this provider to connect

    Server Options:
    - `-m, --model`: Path to GGUF model file
    - `-c, --ctx-size`: Context size (default: 2048)
    - `-ngl, --n-gpu-layers`: Number of layers to offload to GPU
    - `-t, --threads`: Number of CPU threads
    - `--host`: Listen address (default: 127.0.0.1)
    - `--port`: Listen port (default: 8080)

    Supports:
    - Chat completions (/v1/chat/completions)
    - Text completions (/completion)
    - Streaming responses
    - Grammar-based outputs
    - Embeddings (/embedding)
    - Tokenization (/tokenize, /detokenize)

    Example:
        >>> config = LLMConfig(
        ...     provider="llamacpp",
        ...     model="local-model",
        ...     base_url="http://localhost:8080",
        ... )
        >>> async with LlamaCppProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize llama.cpp provider.

        Args:
            config: LLM configuration. base_url defaults to localhost:8080.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._base_url = getattr(config, 'base_url', None) or DEFAULT_BASE_URL

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="llamacpp",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="llama.cpp - CPU/GPU optimized GGUF model inference",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=False,  # Not natively supported
                vision=True,  # LLaVA models supported
                json_mode=True,  # Via grammar
                embeddings=True,
                max_context_tokens=128000,  # Model dependent
                max_output_tokens=4096,
            ),
            supported_models=["any GGUF model"],
            website="https://github.com/ggerganov/llama.cpp",
            documentation="https://github.com/ggerganov/llama.cpp/tree/master/examples/server",
        )

    async def initialize(self) -> None:
        """Initialize the llama.cpp client."""
        if self._initialized:
            return

        httpx = _get_httpx()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout if self._config else 600,  # Long timeout for CPU inference
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the llama.cpp client."""
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
        Generate a completion using llama.cpp server.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional llama.cpp-specific parameters

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Use OpenAI-compatible chat completions endpoint
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
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        # llama.cpp specific parameters
        for key in [
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "presence_penalty",
            "frequency_penalty",
            "mirostat",
            "mirostat_tau",
            "mirostat_eta",
            "seed",
            "grammar",  # For structured outputs
            "json_schema",  # For JSON mode
        ]:
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            response = await self._client.post(
                "/v1/chat/completions",
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
                provider="llamacpp",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("llamacpp", str(e)) from e
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
        """Stream a completion using llama.cpp server."""
        self._ensure_initialized()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        payload = {
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": True,
        }

        if stop:
            payload["stop"] = stop

        for key in ["top_p", "top_k", "repeat_penalty", "seed", "grammar"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
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
            raise ProviderError("llamacpp", str(e)) from e

    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a raw text completion using llama.cpp /completion endpoint.

        This is a direct text completion without chat formatting.

        Args:
            prompt: Full text prompt (including any formatting)
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
            "prompt": prompt,
            "temperature": temp,
            "n_predict": max_tok,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        for key in ["top_p", "top_k", "repeat_penalty", "seed", "grammar"]:
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            response = await self._client.post(
                "/completion",
                json=payload,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()

            usage = TokenUsage(
                prompt_tokens=data.get("tokens_evaluated", 0),
                completion_tokens=data.get("tokens_predicted", 0),
                total_tokens=data.get("tokens_evaluated", 0) + data.get("tokens_predicted", 0),
            )

            return LLMResponse(
                content=data.get("content", ""),
                model=data.get("model", self._model),
                provider="llamacpp",
                usage=usage,
                finish_reason=data.get("stop_type", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("llamacpp", str(e)) from e
            raise

    async def embed(
        self,
        texts: List[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        """
        Generate embeddings using llama.cpp server.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        self._ensure_initialized()

        embeddings = []
        for text in texts:
            try:
                response = await self._client.post(
                    "/embedding",
                    json={"content": text}
                )

                if response.status_code != 200:
                    self._handle_http_error(response)

                data = response.json()
                embeddings.append(data.get("embedding", []))

            except Exception as e:
                raise ProviderError("llamacpp", f"Embedding failed: {e}") from e

        return embeddings

    async def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using llama.cpp server.

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        self._ensure_initialized()

        try:
            response = await self._client.post(
                "/tokenize",
                json={"content": text}
            )

            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            return data.get("tokens", [])

        except Exception as e:
            raise ProviderError("llamacpp", f"Tokenization failed: {e}") from e

    async def detokenize(self, tokens: List[int]) -> str:
        """
        Detokenize tokens using llama.cpp server.

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text
        """
        self._ensure_initialized()

        try:
            response = await self._client.post(
                "/detokenize",
                json={"tokens": tokens}
            )

            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            return data.get("content", "")

        except Exception as e:
            raise ProviderError("llamacpp", f"Detokenization failed: {e}") from e

    async def get_health(self) -> Dict[str, Any]:
        """
        Get server health status.

        Returns:
            Health status dictionary
        """
        self._ensure_initialized()

        try:
            response = await self._client.get("/health")
            return response.json() if response.status_code == 200 else {"status": "error"}
        except Exception:
            return {"status": "unreachable"}

    def _handle_http_error(self, response) -> None:
        """Handle HTTP error responses."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 404:
            raise InvalidModelError(self._model, "llamacpp")
        elif status == 400:
            raise InvalidResponseError("llamacpp", error_msg)
        else:
            raise ProviderError("llamacpp", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("llamacpp", ProviderType.LLM, LlamaCppProvider)
