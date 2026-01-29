"""
ExLlamaV2 LLM Provider

Implements the BaseLLMProvider interface for ExLlamaV2.
Ultra-fast GPTQ/EXL2 quantized model inference with TabbyAPI server.

Official Documentation: https://github.com/turboderp/exllamav2
TabbyAPI Documentation: https://github.com/theroyallab/tabbyAPI
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
                "httpx package is required for ExLlamaV2 provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default settings
DEFAULT_BASE_URL = "http://localhost:5000"
DEFAULT_MODEL = "local-model"


class ExLlamaV2Provider(BaseLLMProvider):
    """
    ExLlamaV2 LLM Provider (via TabbyAPI).

    Provides ultra-fast local inference for quantized models:
    - EXL2 quantization (best quality/speed)
    - GPTQ quantization
    - Speculative decoding
    - Flash Attention 2
    - Paged KV cache

    TabbyAPI Setup (recommended):
    1. Clone TabbyAPI: git clone https://github.com/theroyallab/tabbyAPI
    2. Install: pip install -e .
    3. Configure config.yml with model path
    4. Start: python main.py
    5. Default: http://localhost:5000

    Direct ExLlamaV2 Setup:
    1. Install: pip install exllamav2
    2. Use the examples/chat_server.py from the repo
    3. Or integrate directly with TabbyAPI

    Supports:
    - Chat completions (OpenAI-compatible)
    - Streaming responses
    - Template-based chat formatting
    - Dynamic batching
    - Grammar-based outputs
    - Lora adapters

    Example:
        >>> config = LLMConfig(
        ...     provider="exllamav2",
        ...     model="local-model",
        ...     base_url="http://localhost:5000",
        ... )
        >>> async with ExLlamaV2Provider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize ExLlamaV2 provider.

        Args:
            config: LLM configuration. base_url defaults to localhost:5000.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._base_url = getattr(config, 'base_url', None) or DEFAULT_BASE_URL
        self._api_key = config.api_key if config else None  # TabbyAPI can use auth

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="exllamav2",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="ExLlamaV2 - Ultra-fast GPTQ/EXL2 quantized inference",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=False,
                vision=False,  # Text-only
                json_mode=True,  # Via grammar
                embeddings=False,
                max_context_tokens=32768,  # Model dependent
                max_output_tokens=4096,
            ),
            supported_models=["GPTQ", "EXL2 quantized models"],
            website="https://github.com/turboderp/exllamav2",
            documentation="https://github.com/theroyallab/tabbyAPI",
        )

    async def initialize(self) -> None:
        """Initialize the ExLlamaV2 client."""
        if self._initialized:
            return

        httpx = _get_httpx()

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._config.timeout if self._config else 600,
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the ExLlamaV2 client."""
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
        Generate a completion using ExLlamaV2/TabbyAPI.

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
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        # ExLlamaV2/TabbyAPI specific parameters
        for key in [
            "top_p",
            "top_k",
            "min_p",
            "typical_p",
            "repetition_penalty",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "grammar",
            "grammar_string",
            "token_healing",
            "add_bos_token",
            "ban_eos_token",
            "skip_special_tokens",
            "logit_bias",
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
                provider="exllamav2",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("exllamav2", str(e)) from e
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
        """Stream a completion using ExLlamaV2/TabbyAPI."""
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

        for key in ["top_p", "top_k", "min_p", "repetition_penalty", "seed"]:
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
            raise ProviderError("exllamav2", str(e)) from e

    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a raw text completion.

        Args:
            prompt: Full text prompt
            temperature: Sampling temperature
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
            "max_tokens": max_tok,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        try:
            response = await self._client.post(
                "/v1/completions",
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
                content=choice.get("text", ""),
                model=data.get("model", self._model),
                provider="exllamav2",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("exllamav2", str(e)) from e
            raise

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from TabbyAPI.

        Returns:
            List of model information dictionaries
        """
        self._ensure_initialized()

        try:
            response = await self._client.get("/v1/models")

            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            return data.get("data", [])

        except Exception as e:
            raise ProviderError("exllamav2", f"Failed to list models: {e}") from e

    async def load_model(
        self,
        model_name: str,
        **kwargs: Any,
    ) -> bool:
        """
        Load a specific model in TabbyAPI.

        Args:
            model_name: Name of the model to load
            **kwargs: Additional loading parameters (max_seq_len, cache_size, etc.)

        Returns:
            True if successful
        """
        self._ensure_initialized()

        payload = {"name": model_name}
        payload.update(kwargs)

        try:
            response = await self._client.post(
                "/v1/model/load",
                json=payload
            )
            return response.status_code == 200
        except Exception:
            return False

    async def unload_model(self) -> bool:
        """
        Unload the current model.

        Returns:
            True if successful
        """
        self._ensure_initialized()

        try:
            response = await self._client.post("/v1/model/unload")
            return response.status_code == 200
        except Exception:
            return False

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.

        Returns:
            Model information dictionary
        """
        self._ensure_initialized()

        try:
            response = await self._client.get("/v1/model")
            if response.status_code == 200:
                return response.json()
            return {"error": "No model loaded"}
        except Exception:
            return {"error": "Failed to get model info"}

    async def list_loras(self) -> List[Dict[str, Any]]:
        """
        List available LoRA adapters.

        Returns:
            List of LoRA adapter information
        """
        self._ensure_initialized()

        try:
            response = await self._client.get("/v1/lora/list")
            if response.status_code == 200:
                return response.json().get("data", [])
            return []
        except Exception:
            return []

    async def load_lora(
        self,
        lora_name: str,
        scaling: float = 1.0,
    ) -> bool:
        """
        Load a LoRA adapter.

        Args:
            lora_name: Name of the LoRA adapter
            scaling: LoRA scaling factor

        Returns:
            True if successful
        """
        self._ensure_initialized()

        try:
            response = await self._client.post(
                "/v1/lora/load",
                json={"name": lora_name, "scaling": scaling}
            )
            return response.status_code == 200
        except Exception:
            return False

    def _handle_http_error(self, response) -> None:
        """Handle HTTP error responses."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 404:
            raise InvalidModelError(self._model, "exllamav2")
        elif status == 400:
            raise InvalidResponseError("exllamav2", error_msg)
        else:
            raise ProviderError("exllamav2", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("exllamav2", ProviderType.LLM, ExLlamaV2Provider)
