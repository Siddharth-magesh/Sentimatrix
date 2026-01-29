"""
Text Generation WebUI LLM Provider

Implements the BaseLLMProvider interface for oobabooga's text-generation-webui.
Popular GUI for running local LLMs with multiple backend support.

Official Documentation: https://github.com/oobabooga/text-generation-webui
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
                "httpx package is required for text-generation-webui provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default settings
DEFAULT_BASE_URL = "http://localhost:5000"
DEFAULT_MODEL = "local-model"


class TextGenProvider(BaseLLMProvider):
    """
    Text Generation WebUI LLM Provider.

    Provides local inference through oobabooga's text-generation-webui:
    - Multiple backend support (transformers, llama.cpp, ExLlamaV2, AutoGPTQ)
    - Character/roleplay chat
    - Extensions ecosystem
    - Web GUI + API

    text-generation-webui Setup:
    1. Clone repo: git clone https://github.com/oobabooga/text-generation-webui
    2. Run installer: ./start_linux.sh (or start_windows.bat)
    3. Enable API in settings or use: --api flag
    4. Default API: http://localhost:5000

    Start with API:
        python server.py --api --listen

    With OpenAI-compatible API:
        python server.py --api --extensions openai

    Supports:
    - Chat completions (OpenAI-compatible)
    - Text completions
    - Streaming responses
    - Character/instruction modes
    - Multiple model loading backends

    Example:
        >>> config = LLMConfig(
        ...     provider="textgen",
        ...     model="local-model",
        ...     base_url="http://localhost:5000",
        ... )
        >>> async with TextGenProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize text-generation-webui provider.

        Args:
            config: LLM configuration. base_url defaults to localhost:5000.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._base_url = getattr(config, 'base_url', None) or DEFAULT_BASE_URL
        # Check if using OpenAI-compatible endpoint
        self._openai_mode = getattr(config, 'openai_mode', True)

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="textgen",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="text-generation-webui - Multi-backend local LLM inference",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=False,
                vision=True,  # With multimodal extension
                json_mode=True,
                embeddings=False,  # Not built-in
                max_context_tokens=32768,  # Model dependent
                max_output_tokens=4096,
            ),
            supported_models=["any HuggingFace/GGUF model"],
            website="https://github.com/oobabooga/text-generation-webui",
            documentation="https://github.com/oobabooga/text-generation-webui/wiki",
        )

    async def initialize(self) -> None:
        """Initialize the text-generation-webui client."""
        if self._initialized:
            return

        httpx = _get_httpx()
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Content-Type": "application/json",
            },
            timeout=self._config.timeout if self._config else 600,
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the text-generation-webui client."""
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
        Generate a completion using text-generation-webui.

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

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )

        if self._openai_mode:
            # Use OpenAI-compatible endpoint
            return await self._generate_openai_mode(
                prompt, system_prompt, temp, max_tok, stop, start_time, **kwargs
            )
        else:
            # Use native API
            return await self._generate_native_mode(
                prompt, system_prompt, temp, max_tok, stop, start_time, **kwargs
            )

    async def _generate_openai_mode(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        start_time: float,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate using OpenAI-compatible endpoint."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        for key in ["top_p", "top_k", "repetition_penalty", "seed"]:
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
                provider="textgen",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("textgen", str(e)) from e
            raise

    async def _generate_native_mode(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        start_time: float,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate using native text-generation-webui API."""
        # Build prompt with system message
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "prompt": full_prompt,
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }

        if stop:
            payload["stop_at_newline"] = False
            payload["custom_stopping_strings"] = stop

        # Native API specific parameters
        for key in [
            "top_p",
            "top_k",
            "typical_p",
            "repetition_penalty",
            "encoder_repetition_penalty",
            "no_repeat_ngram_size",
            "seed",
            "truncation_length",
            "add_bos_token",
            "ban_eos_token",
        ]:
            if key in kwargs:
                payload[key] = kwargs[key]

        try:
            response = await self._client.post(
                "/api/v1/generate",
                json=payload,
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            if response.status_code != 200:
                self._handle_http_error(response)

            data = response.json()
            content = data.get("results", [{}])[0].get("text", "")

            usage = TokenUsage(
                prompt_tokens=0,  # Not provided by native API
                completion_tokens=0,
                total_tokens=0,
            )

            return LLMResponse(
                content=content,
                model=self._model,
                provider="textgen",
                usage=usage,
                finish_reason="stop",
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("textgen", str(e)) from e
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
        """Stream a completion using text-generation-webui."""
        self._ensure_initialized()

        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        if self._openai_mode:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload = {
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
                raise ProviderError("textgen", str(e)) from e
        else:
            # Native streaming API
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            payload = {
                "prompt": full_prompt,
                "max_new_tokens": max_tok,
                "temperature": temp,
                "do_sample": temp > 0,
            }

            try:
                async with self._client.stream(
                    "POST",
                    "/api/v1/stream",
                    json=payload,
                ) as response:
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            try:
                                import json
                                data = json.loads(line[6:])
                                if "text" in data:
                                    yield data["text"]
                            except Exception:
                                continue

            except Exception as e:
                raise ProviderError("textgen", str(e)) from e

    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the currently loaded model.

        Returns:
            Model information dictionary
        """
        self._ensure_initialized()

        try:
            response = await self._client.get("/api/v1/model")
            if response.status_code == 200:
                return response.json()
            return {"model": "unknown"}
        except Exception:
            return {"model": "unknown", "error": "Failed to get model info"}

    async def list_models(self) -> List[str]:
        """
        List available models.

        Returns:
            List of model names
        """
        self._ensure_initialized()

        try:
            if self._openai_mode:
                response = await self._client.get("/v1/models")
            else:
                response = await self._client.get("/api/v1/model/list")

            if response.status_code == 200:
                data = response.json()
                if self._openai_mode:
                    return [m["id"] for m in data.get("data", [])]
                return data.get("model_names", [])
            return []
        except Exception:
            return []

    async def load_model(self, model_name: str, **kwargs) -> bool:
        """
        Load a specific model.

        Args:
            model_name: Name of the model to load
            **kwargs: Additional loading parameters

        Returns:
            True if successful
        """
        self._ensure_initialized()

        try:
            payload = {
                "model_name": model_name,
                **kwargs
            }
            response = await self._client.post(
                "/api/v1/model/load",
                json=payload
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
            raise InvalidModelError(self._model, "textgen")
        elif status == 400:
            raise InvalidResponseError("textgen", error_msg)
        else:
            raise ProviderError("textgen", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("textgen", ProviderType.LLM, TextGenProvider)
