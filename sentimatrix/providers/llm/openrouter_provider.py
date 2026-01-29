"""
OpenRouter LLM Provider

Implements the BaseLLMProvider interface for OpenRouter.
Unified API gateway to 200+ models from multiple providers.

Official API Documentation: https://openrouter.ai/docs
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
                "httpx package is required for OpenRouter provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default model
DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

# Popular model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Anthropic
    "anthropic/claude-3.5-sonnet": {
        "context": 200000, "vision": True, "provider": "anthropic"
    },
    "anthropic/claude-3.5-haiku": {
        "context": 200000, "vision": True, "provider": "anthropic"
    },
    "anthropic/claude-3-opus": {
        "context": 200000, "vision": True, "provider": "anthropic"
    },
    # OpenAI
    "openai/gpt-4o": {
        "context": 128000, "vision": True, "provider": "openai"
    },
    "openai/gpt-4o-mini": {
        "context": 128000, "vision": True, "provider": "openai"
    },
    "openai/o1-preview": {
        "context": 128000, "vision": False, "provider": "openai"
    },
    "openai/o1-mini": {
        "context": 128000, "vision": False, "provider": "openai"
    },
    # Google
    "google/gemini-2.0-flash-exp:free": {
        "context": 1000000, "vision": True, "provider": "google"
    },
    "google/gemini-pro-1.5": {
        "context": 2000000, "vision": True, "provider": "google"
    },
    # Meta Llama
    "meta-llama/llama-3.3-70b-instruct": {
        "context": 128000, "vision": False, "provider": "meta"
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "context": 128000, "vision": False, "provider": "meta"
    },
    # Mistral
    "mistralai/mistral-large-2411": {
        "context": 128000, "vision": False, "provider": "mistral"
    },
    "mistralai/pixtral-large-2411": {
        "context": 128000, "vision": True, "provider": "mistral"
    },
    # DeepSeek
    "deepseek/deepseek-chat": {
        "context": 64000, "vision": False, "provider": "deepseek"
    },
    "deepseek/deepseek-r1": {
        "context": 64000, "vision": False, "provider": "deepseek"
    },
    # Qwen
    "qwen/qwq-32b-preview": {
        "context": 32000, "vision": False, "provider": "qwen"
    },
    "qwen/qwen-2.5-72b-instruct": {
        "context": 128000, "vision": False, "provider": "qwen"
    },
    # xAI
    "x-ai/grok-2-1212": {
        "context": 131072, "vision": False, "provider": "xai"
    },
    # Free models
    "google/gemini-2.0-flash-thinking-exp:free": {
        "context": 40000, "vision": True, "provider": "google"
    },
    "meta-llama/llama-3.2-90b-vision-instruct:free": {
        "context": 128000, "vision": True, "provider": "meta"
    },
}


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter LLM Provider.

    Unified API gateway to access 200+ models from:
    - Anthropic (Claude)
    - OpenAI (GPT-4, o1)
    - Google (Gemini)
    - Meta (Llama)
    - Mistral
    - DeepSeek
    - And many more

    Supports:
    - Chat completions
    - Streaming responses
    - Function/tool calling
    - Vision (model-dependent)
    - Automatic fallback routing
    - Cost tracking

    Example:
        >>> config = LLMConfig(
        ...     provider="openrouter",
        ...     api_key="your-api-key",
        ...     model="anthropic/claude-3.5-sonnet",
        ... )
        >>> async with OpenRouterProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    API_BASE = "https://openrouter.ai/api/v1"

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize OpenRouter provider.

        Args:
            config: LLM configuration with OpenRouter API key.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._api_key = config.api_key if config else None
        self._site_url = getattr(config, 'site_url', None) or "https://sentimatrix.dev"
        self._site_name = getattr(config, 'site_name', None) or "Sentimatrix"

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model_config = MODEL_CONFIGS.get(self._model, {})
        context_limit = model_config.get("context", 128000)

        return ProviderInfo(
            name="openrouter",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="OpenRouter - Unified gateway to 200+ models",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=True,
                vision=model_config.get("vision", False),
                json_mode=True,
                embeddings=False,
                max_context_tokens=context_limit,
                max_output_tokens=16384,
            ),
            supported_models=list(MODEL_CONFIGS.keys()),
            website="https://openrouter.ai",
            documentation="https://openrouter.ai/docs",
        )

    async def initialize(self) -> None:
        """Initialize the OpenRouter client."""
        if self._initialized:
            return

        api_key = self._api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "openrouter",
                "API key not provided. Set OPENROUTER_API_KEY environment variable "
                "or pass api_key in config."
            )

        httpx = _get_httpx()
        self._client = httpx.AsyncClient(
            base_url=self.API_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self._site_url,
                "X-Title": self._site_name,
            },
            timeout=self._config.timeout if self._config else 120,
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the OpenRouter client."""
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
        Generate a completion using OpenRouter.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
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

        # OpenRouter-specific options
        if "route" in kwargs:
            payload["route"] = kwargs["route"]

        if "transforms" in kwargs:
            payload["transforms"] = kwargs["transforms"]

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
                provider="openrouter",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("openrouter", str(e)) from e
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
        """Stream a completion using OpenRouter."""
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
            raise ProviderError("openrouter", str(e)) from e

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from OpenRouter.

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
            raise ProviderError("openrouter", f"Failed to list models: {e}") from e

    def _handle_http_error(self, response) -> None:
        """Handle HTTP error responses."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 401:
            raise AuthenticationError("openrouter", error_msg)
        elif status == 429:
            raise RateLimitError("openrouter", retry_after=60)
        elif status == 400:
            if "content" in error_msg.lower() and "filter" in error_msg.lower():
                raise ContentFilteredError("openrouter", error_msg)
            elif "token" in error_msg.lower():
                raise TokenLimitExceededError("openrouter", 0, 0)
            raise InvalidResponseError("openrouter", error_msg)
        elif status == 404:
            raise InvalidModelError(self._model, "openrouter")
        else:
            raise ProviderError("openrouter", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("openrouter", ProviderType.LLM, OpenRouterProvider)
