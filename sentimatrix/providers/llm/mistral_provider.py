"""
Mistral AI LLM Provider

Implements the BaseLLMProvider interface for Mistral AI.
European AI provider with state-of-the-art open and commercial models.

Official API Documentation: https://docs.mistral.ai/
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
_mistralai = None
_httpx = None


def _get_mistralai():
    """Lazy import of mistralai module."""
    global _mistralai
    if _mistralai is None:
        try:
            import mistralai
            _mistralai = mistralai
        except ImportError:
            raise ImportError(
                "mistralai package is required for Mistral provider. "
                "Install it with: pip install mistralai"
            )
    return _mistralai


def _get_httpx():
    """Lazy import of httpx for fallback."""
    global _httpx
    if _httpx is None:
        try:
            import httpx
            _httpx = httpx
        except ImportError:
            raise ImportError(
                "httpx package is required for Mistral provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default model
DEFAULT_MODEL = "mistral-large-latest"

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Premier models
    "mistral-large-latest": {
        "context": 128000, "vision": False, "function_calling": True,
        "description": "Top-tier reasoning model"
    },
    "mistral-large-2411": {
        "context": 128000, "vision": False, "function_calling": True,
        "description": "Mistral Large November 2024"
    },
    "pixtral-large-latest": {
        "context": 128000, "vision": True, "function_calling": True,
        "description": "Multimodal with vision"
    },
    # Free models
    "mistral-small-latest": {
        "context": 32000, "vision": False, "function_calling": True,
        "description": "Cost-efficient for simple tasks"
    },
    "ministral-8b-latest": {
        "context": 128000, "vision": False, "function_calling": True,
        "description": "Edge-optimized 8B model"
    },
    "ministral-3b-latest": {
        "context": 128000, "vision": False, "function_calling": True,
        "description": "Edge-optimized 3B model"
    },
    # Specialized
    "codestral-latest": {
        "context": 32000, "vision": False, "function_calling": False,
        "description": "Code generation specialist"
    },
    "mistral-embed": {
        "context": 8192, "vision": False, "function_calling": False,
        "description": "Text embeddings"
    },
    # Legacy
    "mistral-medium": {
        "context": 32000, "vision": False, "function_calling": True,
        "description": "Legacy medium model"
    },
    "open-mistral-7b": {
        "context": 32000, "vision": False, "function_calling": False,
        "description": "Open source 7B"
    },
    "open-mixtral-8x7b": {
        "context": 32000, "vision": False, "function_calling": False,
        "description": "Open source Mixtral"
    },
    "open-mixtral-8x22b": {
        "context": 64000, "vision": False, "function_calling": True,
        "description": "Open source Mixtral 8x22B"
    },
}


class MistralProvider(BaseLLMProvider):
    """
    Mistral AI LLM Provider.

    Provides access to Mistral's state-of-the-art models:
    - Mistral Large (top-tier reasoning)
    - Pixtral Large (multimodal with vision)
    - Mistral Small (cost-efficient)
    - Codestral (code generation)
    - Open source models (Mistral 7B, Mixtral)

    Supports:
    - Chat completions
    - Streaming responses
    - Function/tool calling
    - Vision (Pixtral models)
    - Text embeddings
    - JSON mode

    Example:
        >>> config = LLMConfig(
        ...     provider="mistral",
        ...     api_key="your-api-key",
        ...     model="mistral-large-latest",
        ... )
        >>> async with MistralProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Mistral provider.

        Args:
            config: LLM configuration with Mistral API key.
        """
        super().__init__(config)
        self._client: Any = None
        self._async_client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._api_key = config.api_key if config else None
        self._api_base = config.api_base if config else "https://api.mistral.ai/v1"

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model_config = MODEL_CONFIGS.get(self._model, {})
        context_limit = model_config.get("context", 32000)

        return ProviderInfo(
            name="mistral",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Mistral AI - European AI leader",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=model_config.get("function_calling", True),
                vision=model_config.get("vision", False),
                json_mode=True,
                embeddings=True,
                max_context_tokens=context_limit,
                max_output_tokens=min(8192, context_limit),
            ),
            supported_models=list(MODEL_CONFIGS.keys()),
            website="https://mistral.ai",
            documentation="https://docs.mistral.ai/",
        )

    async def initialize(self) -> None:
        """Initialize the Mistral client."""
        if self._initialized:
            return

        api_key = self._api_key or os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "mistral",
                "API key not provided. Set MISTRAL_API_KEY environment variable "
                "or pass api_key in config."
            )

        try:
            mistralai = _get_mistralai()
            self._client = mistralai.Mistral(api_key=api_key)
            self._initialized = True
        except Exception:
            # Fallback to httpx if mistralai package not available
            httpx = _get_httpx()
            self._async_client = httpx.AsyncClient(
                base_url=self._api_base,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self._config.timeout if self._config else 60,
            )
            self._initialized = True

    async def close(self) -> None:
        """Close the Mistral client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
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
        Generate a completion using Mistral AI.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters (response_format, tools, etc.)

        Returns:
            LLMResponse with generated content
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request parameters
        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )

        try:
            if self._client:
                # Use official SDK
                response = await self._generate_with_sdk(
                    messages, temp, max_tok, stop, **kwargs
                )
            else:
                # Use httpx fallback
                response = await self._generate_with_httpx(
                    messages, temp, max_tok, stop, **kwargs
                )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            return LLMResponse(
                content=response["content"],
                model=response.get("model", self._model),
                provider="mistral",
                usage=response["usage"],
                finish_reason=response.get("finish_reason", "stop"),
                response_time_ms=elapsed_ms,
                raw_response=response.get("raw"),
            )

        except Exception as e:
            self._handle_error(e)

    async def _generate_with_sdk(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate using official Mistral SDK."""
        import asyncio

        params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop:
            params["stop"] = stop

        if "response_format" in kwargs:
            params["response_format"] = kwargs["response_format"]

        if "tools" in kwargs:
            params["tools"] = kwargs["tools"]

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.chat.complete(**params)
        )

        choice = response.choices[0]
        usage = TokenUsage(
            prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
            completion_tokens=response.usage.completion_tokens if response.usage else 0,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )

        return {
            "content": choice.message.content or "",
            "model": response.model,
            "usage": usage,
            "finish_reason": choice.finish_reason,
            "raw": response,
        }

    async def _generate_with_httpx(
        self,
        messages: List[Dict],
        temperature: float,
        max_tokens: int,
        stop: Optional[List[str]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate using httpx fallback."""
        payload = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if stop:
            payload["stop"] = stop

        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        response = await self._async_client.post(
            "/chat/completions",
            json=payload,
        )

        if response.status_code != 200:
            raise ProviderError("mistral", f"API error: {response.text}")

        data = response.json()
        choice = data["choices"][0]
        usage_data = data.get("usage", {})

        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        return {
            "content": choice["message"]["content"] or "",
            "model": data.get("model", self._model),
            "usage": usage,
            "finish_reason": choice.get("finish_reason", "stop"),
            "raw": data,
        }

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion using Mistral AI."""
        self._ensure_initialized()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        if self._client:
            # Use SDK streaming
            import asyncio
            loop = asyncio.get_event_loop()

            params = {
                "model": self._model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok,
            }

            if stop:
                params["stop"] = stop

            stream = await loop.run_in_executor(
                None,
                lambda: self._client.chat.stream(**params)
            )

            for chunk in stream:
                if chunk.data.choices:
                    delta = chunk.data.choices[0].delta
                    if delta.content:
                        yield delta.content
        else:
            # Use httpx streaming
            payload = {
                "model": self._model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": max_tok,
                "stream": True,
            }

            if stop:
                payload["stop"] = stop

            async with self._async_client.stream(
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

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embeddings using Mistral AI.

        Args:
            texts: List of texts to embed
            model: Embedding model (default: mistral-embed)

        Returns:
            List of embedding vectors
        """
        self._ensure_initialized()

        embed_model = model or "mistral-embed"

        if self._client:
            import asyncio
            loop = asyncio.get_event_loop()

            response = await loop.run_in_executor(
                None,
                lambda: self._client.embeddings.create(
                    model=embed_model,
                    inputs=texts,
                )
            )

            return [item.embedding for item in response.data]
        else:
            response = await self._async_client.post(
                "/embeddings",
                json={
                    "model": embed_model,
                    "input": texts,
                }
            )

            if response.status_code != 200:
                raise ProviderError("mistral", f"Embedding error: {response.text}")

            data = response.json()
            return [item["embedding"] for item in data["data"]]

    def _handle_error(self, error: Exception) -> None:
        """Handle Mistral API errors."""
        error_msg = str(error)

        if "401" in error_msg or "Unauthorized" in error_msg:
            raise AuthenticationError("mistral", error_msg) from error
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise RateLimitError("mistral", retry_after=60) from error
        elif "400" in error_msg:
            if "content" in error_msg.lower() and "filter" in error_msg.lower():
                raise ContentFilteredError("mistral", error_msg) from error
            elif "token" in error_msg.lower():
                raise TokenLimitExceededError("mistral", 0, 0) from error
            raise InvalidResponseError("mistral", error_msg) from error
        elif "404" in error_msg or "model" in error_msg.lower():
            raise InvalidModelError(self._model, "mistral") from error
        else:
            raise ProviderError("mistral", error_msg) from error


# Register the provider
register_provider("mistral", ProviderType.LLM, MistralProvider)
