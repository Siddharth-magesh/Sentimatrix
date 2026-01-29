"""
DeepSeek LLM Provider

Implements the BaseLLMProvider interface for DeepSeek.
State-of-the-art reasoning and coding models from China.

Official API Documentation: https://platform.deepseek.com/api-docs
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
                "httpx package is required for DeepSeek provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default model
DEFAULT_MODEL = "deepseek-chat"

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "deepseek-chat": {
        "context": 64000,
        "vision": False,
        "description": "DeepSeek V3 - Advanced conversational AI"
    },
    "deepseek-coder": {
        "context": 64000,
        "vision": False,
        "description": "DeepSeek Coder - Specialized for code"
    },
    "deepseek-reasoner": {
        "context": 64000,
        "vision": False,
        "description": "DeepSeek R1 - Advanced reasoning model"
    },
}


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek LLM Provider.

    State-of-the-art AI models from DeepSeek:
    - DeepSeek V3 (deepseek-chat): Advanced conversational model
    - DeepSeek Coder: Specialized code generation
    - DeepSeek R1 (deepseek-reasoner): Advanced reasoning

    Key Features:
    - Competitive with GPT-4/Claude on benchmarks
    - Excellent coding capabilities
    - Strong math and reasoning
    - Cost-effective pricing
    - OpenAI-compatible API

    Supports:
    - Chat completions
    - Streaming responses
    - Function/tool calling
    - JSON mode

    Example:
        >>> config = LLMConfig(
        ...     provider="deepseek",
        ...     api_key="your-api-key",
        ...     model="deepseek-chat",
        ... )
        >>> async with DeepSeekProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    API_BASE = "https://api.deepseek.com/v1"

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize DeepSeek provider.

        Args:
            config: LLM configuration with DeepSeek API key.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._api_key = config.api_key if config else None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model_config = MODEL_CONFIGS.get(self._model, {})
        context_limit = model_config.get("context", 64000)

        return ProviderInfo(
            name="deepseek",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="DeepSeek - State-of-the-art reasoning and coding",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=True,
                vision=False,
                json_mode=True,
                embeddings=False,
                max_context_tokens=context_limit,
                max_output_tokens=8192,
            ),
            supported_models=list(MODEL_CONFIGS.keys()),
            website="https://deepseek.com",
            documentation="https://platform.deepseek.com/api-docs",
        )

    async def initialize(self) -> None:
        """Initialize the DeepSeek client."""
        if self._initialized:
            return

        api_key = self._api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "deepseek",
                "API key not provided. Set DEEPSEEK_API_KEY environment variable "
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
        """Close the DeepSeek client."""
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
        Generate a completion using DeepSeek.

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

            # Handle reasoning tokens if present (DeepSeek R1)
            if "reasoning_tokens" in usage_data:
                usage.reasoning_tokens = usage_data["reasoning_tokens"]

            return LLMResponse(
                content=choice["message"]["content"] or "",
                model=data.get("model", self._model),
                provider="deepseek",
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            if not isinstance(e, ProviderError):
                raise ProviderError("deepseek", str(e)) from e
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
        """Stream a completion using DeepSeek."""
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
            raise ProviderError("deepseek", str(e)) from e

    async def generate_with_reasoning(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate with reasoning chain (DeepSeek R1).

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Dictionary with content, reasoning, and usage
        """
        # Use deepseek-reasoner model for reasoning
        original_model = self._model
        self._model = "deepseek-reasoner"

        try:
            response = await self.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )

            # Parse reasoning from response if available
            return {
                "content": response.content,
                "reasoning": response.raw_response.get("reasoning", ""),
                "usage": response.usage,
                "model": response.model,
            }

        finally:
            self._model = original_model

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from DeepSeek.

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
            raise ProviderError("deepseek", f"Failed to list models: {e}") from e

    def _handle_http_error(self, response) -> None:
        """Handle HTTP error responses."""
        status = response.status_code
        try:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_msg = response.text

        if status == 401:
            raise AuthenticationError("deepseek", error_msg)
        elif status == 429:
            raise RateLimitError("deepseek", retry_after=60)
        elif status == 400:
            if "token" in error_msg.lower():
                raise TokenLimitExceededError("deepseek", 0, 0)
            raise InvalidResponseError("deepseek", error_msg)
        elif status == 404:
            raise InvalidModelError(self._model, "deepseek")
        else:
            raise ProviderError("deepseek", f"HTTP {status}: {error_msg}")


# Register the provider
register_provider("deepseek", ProviderType.LLM, DeepSeekProvider)
