"""
OpenAI LLM Provider

Implements the BaseLLMProvider interface for OpenAI's API.
Supports GPT-4o, GPT-4o-mini, o1, and embedding models.

Official API Documentation: https://platform.openai.com/docs/api-reference
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    AuthenticationError,
    ContentFilteredError,
    ErrorCode,
    InvalidModelError,
    InvalidResponseError,
    OpenAIError,
    RateLimitError,
    TokenLimitExceededError,
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

# Lazy imports for optional dependencies
_openai = None
_tiktoken = None


def _get_openai():
    """Lazy import of openai module."""
    global _openai
    if _openai is None:
        try:
            import openai
            _openai = openai
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAI provider. "
                "Install it with: pip install openai"
            )
    return _openai


def _get_tiktoken():
    """Lazy import of tiktoken module."""
    global _tiktoken
    if _tiktoken is None:
        try:
            import tiktoken
            _tiktoken = tiktoken
        except ImportError:
            _tiktoken = None  # Optional, will fall back to estimate
    return _tiktoken


# Default models for different use cases
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Model context limits
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-4-turbo-preview": 128000,
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-3.5-turbo": 16385,
    "gpt-3.5-turbo-16k": 16385,
    "o1-preview": 128000,
    "o1-mini": 128000,
}

# Models that support specific features
VISION_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview"}
FUNCTION_CALLING_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview",
    "gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
}
JSON_MODE_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4-turbo-preview"}


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI LLM Provider.

    Supports:
    - Chat completions (GPT-4o, GPT-4o-mini, GPT-3.5-turbo)
    - Streaming responses
    - Function/tool calling
    - JSON mode
    - Vision (image inputs)
    - Text embeddings

    Example:
        >>> config = LLMConfig(
        ...     provider="openai",
        ...     model="gpt-4o-mini",
        ...     api_key="sk-..."
        ... )
        >>> async with OpenAIProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize OpenAI provider.

        Args:
            config: LLM configuration. If not provided, uses defaults.
        """
        super().__init__(config)
        self._client: Any = None
        self._async_client: Any = None
        self._model = config.model if config else DEFAULT_CHAT_MODEL
        self._api_key = config.api_key if config else None
        self._api_base = config.api_base if config else None
        self._organization = config.organization if config else None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model = self._model
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 128000)

        return ProviderInfo(
            name="openai",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="OpenAI GPT models provider",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=model in FUNCTION_CALLING_MODELS,
                vision=model in VISION_MODELS,
                json_mode=model in JSON_MODE_MODELS,
                embeddings=True,
                max_context_tokens=context_limit,
                max_output_tokens=min(16384, context_limit),
            ),
            supported_models=list(MODEL_CONTEXT_LIMITS.keys()),
            website="https://openai.com",
            documentation="https://platform.openai.com/docs",
        )

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if self._initialized:
            return

        openai = _get_openai()

        # Resolve API key
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "openai",
                "API key not provided. Set OPENAI_API_KEY environment variable or pass api_key in config."
            )

        # Create async client
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": self._config.timeout if self._config else 30,
        }

        if self._api_base:
            client_kwargs["base_url"] = self._api_base

        if self._organization:
            client_kwargs["organization"] = self._organization

        self._async_client = openai.AsyncOpenAI(**client_kwargs)
        self._client = openai.OpenAI(**client_kwargs)
        self._initialized = True

    async def close(self) -> None:
        """Close the OpenAI client."""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None
        if self._client:
            self._client.close()
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
        Generate a completion using OpenAI's Chat API.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters (response_format, etc.)

        Returns:
            LLMResponse with generated content

        Raises:
            OpenAIError: On API errors
            AuthenticationError: On authentication failure
            RateLimitError: On rate limit exceeded
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Build messages
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        # Apply optional parameters
        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        params["temperature"] = temp

        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )
        params["max_tokens"] = max_tok

        if stop:
            params["stop"] = stop

        # Handle response_format for JSON mode
        if "response_format" in kwargs and self._model in JSON_MODE_MODELS:
            params["response_format"] = kwargs.pop("response_format")

        # Handle additional kwargs
        for key in ["top_p", "frequency_penalty", "presence_penalty", "seed"]:
            if key in kwargs:
                params[key] = kwargs.pop(key)

        try:
            response = await self._async_client.chat.completions.create(**params)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            # Build usage stats
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return LLMResponse(
                content=content,
                model=response.model,
                provider="openai",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_error(e)

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion using OpenAI's Chat API.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Yields:
            Text chunks as they're generated

        Raises:
            OpenAIError: On API errors
        """
        self._ensure_initialized()

        # Build messages
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        params["temperature"] = temp

        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )
        params["max_tokens"] = max_tok

        if stop:
            params["stop"] = stop

        try:
            stream = await self._async_client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            self._handle_error(e)

    async def generate_with_functions(
        self,
        prompt: str,
        functions: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        function_call: Union[str, Dict[str, str]] = "auto",
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion with function calling.

        Args:
            prompt: User message
            functions: List of function definitions (OpenAI tools format)
            system_prompt: Optional system message
            function_call: "auto", "none", or {"type": "function", "function": {"name": "..."}}
            **kwargs: Additional parameters

        Returns:
            LLMResponse with potential tool_calls

        Example:
            >>> functions = [{
            ...     "type": "function",
            ...     "function": {
            ...         "name": "get_weather",
            ...         "description": "Get the weather in a location",
            ...         "parameters": {
            ...             "type": "object",
            ...             "properties": {
            ...                 "location": {"type": "string"}
            ...             },
            ...             "required": ["location"]
            ...         }
            ...     }
            ... }]
            >>> response = await provider.generate_with_functions(
            ...     "What's the weather in Paris?",
            ...     functions
            ... )
        """
        if self._model not in FUNCTION_CALLING_MODELS:
            raise InvalidModelError("openai", self._model)

        self._ensure_initialized()

        start_time = time.perf_counter()

        # Build messages
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": functions,
        }

        # Handle tool_choice
        if function_call == "auto":
            params["tool_choice"] = "auto"
        elif function_call == "none":
            params["tool_choice"] = "none"
        elif isinstance(function_call, dict):
            params["tool_choice"] = function_call
        elif isinstance(function_call, str):
            params["tool_choice"] = {"type": "function", "function": {"name": function_call}}

        temp = kwargs.get("temperature", self._config.temperature if self._config else 0.7)
        params["temperature"] = temp

        max_tok = kwargs.get("max_tokens", self._config.max_tokens if self._config else 1024)
        params["max_tokens"] = max_tok

        try:
            response = await self._async_client.chat.completions.create(**params)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            choice = response.choices[0]
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

            # Extract tool calls
            tool_calls = None
            if choice.message.tool_calls:
                tool_calls = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    }
                    for tc in choice.message.tool_calls
                ]

            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
            )

            return LLMResponse(
                content=content,
                model=response.model,
                provider="openai",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                tool_calls=tool_calls,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_error(e)

    async def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text.

        Args:
            text: Single text or list of texts
            model: Embedding model (default: text-embedding-3-small)

        Returns:
            Embedding vector(s)

        Example:
            >>> embedding = await provider.embed("Hello world")
            >>> embeddings = await provider.embed(["Hello", "World"])
        """
        self._ensure_initialized()

        embedding_model = model or DEFAULT_EMBEDDING_MODEL

        # Handle single text vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        try:
            response = await self._async_client.embeddings.create(
                model=embedding_model,
                input=texts,
            )

            embeddings = [item.embedding for item in response.data]

            return embeddings[0] if is_single else embeddings

        except Exception as e:
            self._handle_error(e)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text using tiktoken.

        Args:
            text: Text to count tokens
            model: Model to use for tokenization

        Returns:
            Number of tokens
        """
        tiktoken = _get_tiktoken()

        if tiktoken is None:
            # Fallback to rough estimate
            return len(text) // 4

        model_name = model or self._model

        try:
            encoding = tiktoken.encoding_for_model(model_name)
            return len(encoding.encode(text))
        except KeyError:
            # Fallback for unknown models
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

    def _handle_error(self, error: Exception) -> None:
        """
        Handle OpenAI API errors and convert to Sentimatrix exceptions.

        Args:
            error: Exception to handle

        Raises:
            Appropriate Sentimatrix exception
        """
        openai = _get_openai()

        error_message = str(error)

        # Handle OpenAI-specific errors
        if isinstance(error, openai.AuthenticationError):
            raise AuthenticationError("openai", "Invalid API key") from error

        if isinstance(error, openai.RateLimitError):
            raise RateLimitError(
                "Rate limit exceeded",
                provider="openai",
                retry_after=60,  # OpenAI suggests 60s retry
            ) from error

        if isinstance(error, openai.BadRequestError):
            if "context_length_exceeded" in error_message.lower():
                raise TokenLimitExceededError(
                    "openai",
                    self._model,
                    requested=0,  # Not available from error
                    limit=MODEL_CONTEXT_LIMITS.get(self._model, 0),
                ) from error
            if "content_filter" in error_message.lower():
                raise ContentFilteredError("openai") from error
            raise OpenAIError(error_message, self._model, original_error=error)

        if isinstance(error, openai.NotFoundError):
            raise InvalidModelError("openai", self._model) from error

        if isinstance(error, openai.APITimeoutError):
            from sentimatrix.core.exceptions import TimeoutError
            raise TimeoutError(
                f"OpenAI API timeout: {error_message}",
                timeout=self._config.timeout if self._config else 30,
                operation="generate",
            ) from error

        if isinstance(error, openai.APIConnectionError):
            from sentimatrix.core.exceptions import ConnectionTimeoutError
            raise ConnectionTimeoutError(
                "api.openai.com",
                self._config.timeout if self._config else 30,
            ) from error

        # Generic OpenAI error
        if isinstance(error, openai.OpenAIError):
            raise OpenAIError(error_message, self._model, original_error=error)

        # Re-raise unknown errors
        raise OpenAIError(
            f"Unexpected error: {error_message}",
            self._model,
            original_error=error,
        )


# Register the provider
register_provider("openai", ProviderType.LLM, OpenAIProvider)
