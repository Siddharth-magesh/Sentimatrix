"""
Anthropic LLM Provider

Implements the BaseLLMProvider interface for Anthropic's Claude API.
Supports Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku models.

Official API Documentation: https://docs.anthropic.com/en/api/getting-started
"""

from __future__ import annotations

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    AnthropicError,
    AuthenticationError,
    ContentFilteredError,
    ErrorCode,
    InvalidModelError,
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

# Lazy import
_anthropic = None


def _get_anthropic():
    """Lazy import of anthropic module."""
    global _anthropic
    if _anthropic is None:
        try:
            import anthropic
            _anthropic = anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Anthropic provider. "
                "Install it with: pip install anthropic"
            )
    return _anthropic


# Default model
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Available models and their context limits
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # Claude 4
    "claude-sonnet-4-20250514": 200000,
    # Claude 3.5
    "claude-3-5-sonnet-20241022": 200000,
    "claude-3-5-sonnet-20240620": 200000,
    "claude-3-5-haiku-20241022": 200000,
    # Claude 3
    "claude-3-opus-20240229": 200000,
    "claude-3-sonnet-20240229": 200000,
    "claude-3-haiku-20240307": 200000,
    # Aliases
    "claude-sonnet-4": 200000,
    "claude-3-5-sonnet-latest": 200000,
    "claude-3-5-haiku-latest": 200000,
    "claude-3-opus-latest": 200000,
}

# Max output tokens per model
MODEL_MAX_OUTPUT: Dict[str, int] = {
    "claude-sonnet-4-20250514": 16384,
    "claude-3-5-sonnet-20241022": 8192,
    "claude-3-5-sonnet-20240620": 8192,
    "claude-3-5-haiku-20241022": 8192,
    "claude-3-opus-20240229": 4096,
    "claude-3-sonnet-20240229": 4096,
    "claude-3-haiku-20240307": 4096,
}

# All Claude 3+ models support vision
VISION_MODELS = set(MODEL_CONTEXT_LIMITS.keys())

# All Claude 3+ models support function calling (tool use)
FUNCTION_CALLING_MODELS = set(MODEL_CONTEXT_LIMITS.keys())


class AnthropicProvider(BaseLLMProvider):
    """
    Anthropic Claude LLM Provider.

    Features:
    - Claude 3.5 Sonnet, Claude 3 Opus/Sonnet/Haiku
    - 200K context window
    - Streaming support
    - Function calling (tool use)
    - Vision support (image inputs)
    - Extended thinking (Claude 3.5+)

    Example:
        >>> config = LLMConfig(
        ...     provider="anthropic",
        ...     model="claude-3-5-sonnet-20241022",
        ...     api_key="sk-ant-..."
        ... )
        >>> async with AnthropicProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Anthropic provider.

        Args:
            config: LLM configuration. If not provided, uses defaults.
        """
        super().__init__(config)
        self._client: Any = None
        self._async_client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._api_key = config.api_key if config else None
        self._api_base = config.api_base if config else None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model = self._model
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 200000)
        max_output = MODEL_MAX_OUTPUT.get(model, 4096)

        return ProviderInfo(
            name="anthropic",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Anthropic Claude models provider",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=model in FUNCTION_CALLING_MODELS,
                vision=model in VISION_MODELS,
                json_mode=True,  # Via tool use or prompting
                embeddings=False,  # Anthropic doesn't have embedding API
                max_context_tokens=context_limit,
                max_output_tokens=max_output,
            ),
            supported_models=list(MODEL_CONTEXT_LIMITS.keys()),
            website="https://anthropic.com",
            documentation="https://docs.anthropic.com",
        )

    async def initialize(self) -> None:
        """Initialize the Anthropic client."""
        if self._initialized:
            return

        anthropic = _get_anthropic()

        # Resolve API key
        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "anthropic",
                "API key not provided. Set ANTHROPIC_API_KEY environment variable or pass api_key in config."
            )

        # Create client kwargs
        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": self._config.timeout if self._config else 60,
        }

        if self._api_base:
            client_kwargs["base_url"] = self._api_base

        # Create async client
        self._async_client = anthropic.AsyncAnthropic(**client_kwargs)
        self._client = anthropic.Anthropic(**client_kwargs)
        self._initialized = True

    async def close(self) -> None:
        """Close the Anthropic client."""
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
        Generate a completion using Anthropic's Messages API.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content

        Raises:
            AnthropicError: On API errors
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Build messages (Anthropic format)
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        # System prompt is a separate parameter in Anthropic API
        if system_prompt:
            params["system"] = system_prompt

        # Apply optional parameters
        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        # Anthropic temperature is 0-1, clamp if needed
        params["temperature"] = min(1.0, max(0.0, temp))

        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )
        params["max_tokens"] = max_tok

        if stop:
            params["stop_sequences"] = stop

        # Handle additional kwargs
        if "top_p" in kwargs:
            params["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            params["top_k"] = kwargs.pop("top_k")

        try:
            response = await self._async_client.messages.create(**params)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract response data
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            finish_reason = response.stop_reason or "stop"
            if finish_reason == "end_turn":
                finish_reason = "stop"

            # Build usage stats
            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if response.usage else 0
                ),
            )

            return LLMResponse(
                content=content,
                model=response.model,
                provider="anthropic",
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
        Stream a completion using Anthropic's Messages API.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Yields:
            Text chunks as they're generated
        """
        self._ensure_initialized()

        # Build messages
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        if system_prompt:
            params["system"] = system_prompt

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        params["temperature"] = min(1.0, max(0.0, temp))

        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )
        params["max_tokens"] = max_tok

        if stop:
            params["stop_sequences"] = stop

        try:
            async with self._async_client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    yield text

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
        Generate a completion with function calling (tool use).

        Args:
            prompt: User message
            functions: List of tool definitions (Anthropic tools format)
            system_prompt: Optional system message
            function_call: "auto", "any", or specific tool
            **kwargs: Additional parameters

        Returns:
            LLMResponse with potential tool_calls

        Note:
            Anthropic uses a different tool format than OpenAI.
            This method accepts OpenAI-style tools and converts them.

        Example:
            >>> tools = [{
            ...     "name": "get_weather",
            ...     "description": "Get the weather in a location",
            ...     "input_schema": {
            ...         "type": "object",
            ...         "properties": {
            ...             "location": {"type": "string"}
            ...         },
            ...         "required": ["location"]
            ...     }
            ... }]
            >>> response = await provider.generate_with_functions(
            ...     "What's the weather in Paris?",
            ...     tools
            ... )
        """
        if self._model not in FUNCTION_CALLING_MODELS:
            raise InvalidModelError("anthropic", self._model)

        self._ensure_initialized()

        start_time = time.perf_counter()

        # Build messages
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": prompt}
        ]

        # Convert OpenAI-style tools to Anthropic format if needed
        tools = []
        for func in functions:
            if "type" in func and func["type"] == "function":
                # OpenAI format
                tool = {
                    "name": func["function"]["name"],
                    "description": func["function"].get("description", ""),
                    "input_schema": func["function"].get("parameters", {"type": "object", "properties": {}}),
                }
            else:
                # Already Anthropic format or simple format
                tool = func
            tools.append(tool)

        # Build request parameters
        params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
        }

        if system_prompt:
            params["system"] = system_prompt

        # Handle tool_choice
        if function_call == "auto":
            params["tool_choice"] = {"type": "auto"}
        elif function_call == "any":
            params["tool_choice"] = {"type": "any"}
        elif function_call == "none":
            # Don't include tools if none
            del params["tools"]
        elif isinstance(function_call, str):
            params["tool_choice"] = {"type": "tool", "name": function_call}
        elif isinstance(function_call, dict):
            params["tool_choice"] = function_call

        temp = kwargs.get("temperature", self._config.temperature if self._config else 0.7)
        params["temperature"] = min(1.0, max(0.0, temp))

        max_tok = kwargs.get("max_tokens", self._config.max_tokens if self._config else 1024)
        params["max_tokens"] = max_tok

        try:
            response = await self._async_client.messages.create(**params)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract response data
            content = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content += block.text
                elif block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": str(block.input) if not isinstance(block.input, str) else block.input,
                        }
                    })

            finish_reason = response.stop_reason or "stop"
            if finish_reason == "end_turn":
                finish_reason = "stop"
            elif finish_reason == "tool_use":
                finish_reason = "tool_calls"

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if response.usage else 0
                ),
            )

            return LLMResponse(
                content=content,
                model=response.model,
                provider="anthropic",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                tool_calls=tool_calls if tool_calls else None,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_error(e)

    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: Optional[str] = None,
        image_detail: str = "auto",
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion with image inputs.

        Args:
            prompt: User message
            images: List of image URLs or base64-encoded images
            system_prompt: Optional system message
            image_detail: "auto", "low", or "high" (ignored, kept for compatibility)
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content

        Example:
            >>> response = await provider.generate_with_vision(
            ...     "What's in this image?",
            ...     ["https://example.com/image.jpg"]
            ... )
        """
        self._ensure_initialized()

        import base64

        start_time = time.perf_counter()

        # Build content with images
        content_blocks: List[Dict[str, Any]] = []

        for image in images:
            if isinstance(image, bytes):
                # Base64 encode bytes
                image_data = base64.b64encode(image).decode("utf-8")
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",  # Assume JPEG, could detect
                        "data": image_data,
                    }
                })
            elif image.startswith("data:"):
                # Already base64 encoded with data URL
                # Parse data URL
                parts = image.split(",", 1)
                if len(parts) == 2:
                    header, data = parts
                    media_type = header.split(";")[0].replace("data:", "")
                    content_blocks.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": data,
                        }
                    })
            elif image.startswith("http"):
                # URL - Anthropic supports URLs directly
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": image,
                    }
                })
            else:
                # Assume it's base64 data
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image,
                    }
                })

        # Add text prompt
        content_blocks.append({
            "type": "text",
            "text": prompt,
        })

        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": content_blocks}
        ]

        params: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }

        if system_prompt:
            params["system"] = system_prompt

        temp = kwargs.get("temperature", self._config.temperature if self._config else 0.7)
        params["temperature"] = min(1.0, max(0.0, temp))

        max_tok = kwargs.get("max_tokens", self._config.max_tokens if self._config else 1024)
        params["max_tokens"] = max_tok

        try:
            response = await self._async_client.messages.create(**params)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text

            finish_reason = response.stop_reason or "stop"
            if finish_reason == "end_turn":
                finish_reason = "stop"

            usage = TokenUsage(
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(
                    (response.usage.input_tokens + response.usage.output_tokens)
                    if response.usage else 0
                ),
            )

            return LLMResponse(
                content=content,
                model=response.model,
                provider="anthropic",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_error(e)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Estimate tokens in text.

        Args:
            text: Text to count tokens
            model: Model (unused, kept for interface compatibility)

        Returns:
            Estimated number of tokens

        Note:
            Anthropic doesn't provide a public tokenizer.
            This is a rough estimate (~4 chars per token for English).
        """
        # Anthropic doesn't have a public tokenizer
        # Use rough estimate: ~4 characters per token
        return len(text) // 4

    def _handle_error(self, error: Exception) -> None:
        """
        Handle Anthropic API errors and convert to Sentimatrix exceptions.

        Args:
            error: Exception to handle

        Raises:
            Appropriate Sentimatrix exception
        """
        anthropic = _get_anthropic()

        error_message = str(error)

        # Handle Anthropic-specific errors
        if isinstance(error, anthropic.AuthenticationError):
            raise AuthenticationError("anthropic", "Invalid API key") from error

        if isinstance(error, anthropic.RateLimitError):
            raise RateLimitError(
                "Rate limit exceeded",
                provider="anthropic",
                retry_after=60,
            ) from error

        if isinstance(error, anthropic.BadRequestError):
            if "context" in error_message.lower() or "token" in error_message.lower():
                raise TokenLimitExceededError(
                    "anthropic",
                    self._model,
                    requested=0,
                    limit=MODEL_CONTEXT_LIMITS.get(self._model, 0),
                ) from error
            if "content" in error_message.lower():
                raise ContentFilteredError("anthropic") from error
            raise AnthropicError(error_message, self._model, original_error=error)

        if isinstance(error, anthropic.NotFoundError):
            raise InvalidModelError("anthropic", self._model) from error

        if isinstance(error, anthropic.APITimeoutError):
            from sentimatrix.core.exceptions import TimeoutError
            raise TimeoutError(
                f"Anthropic API timeout: {error_message}",
                timeout=self._config.timeout if self._config else 60,
                operation="generate",
            ) from error

        if isinstance(error, anthropic.APIConnectionError):
            from sentimatrix.core.exceptions import ConnectionTimeoutError
            raise ConnectionTimeoutError(
                "api.anthropic.com",
                self._config.timeout if self._config else 60,
            ) from error

        # Generic Anthropic error
        if isinstance(error, anthropic.APIError):
            raise AnthropicError(error_message, self._model, original_error=error)

        # Re-raise unknown errors
        raise AnthropicError(
            f"Unexpected error: {error_message}",
            self._model,
            original_error=error,
        )


# Register the provider
register_provider("anthropic", ProviderType.LLM, AnthropicProvider)
