"""
Groq LLM Provider

Implements the BaseLLMProvider interface for Groq's ultra-fast inference API.
Supports LLaMA, Mixtral, and Gemma models with extremely low latency.

Official API Documentation: https://console.groq.com/docs/api-reference
Free Tier: Available (30 requests/minute, 6000 tokens/minute)
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
    GroqError,
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
_groq = None


def _get_groq():
    """Lazy import of groq module."""
    global _groq
    if _groq is None:
        try:
            import groq
            _groq = groq
        except ImportError:
            raise ImportError(
                "groq package is required for Groq provider. "
                "Install it with: pip install groq"
            )
    return _groq


# Default model
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Available models and their context limits
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # LLaMA 3.3
    "llama-3.3-70b-versatile": 128000,
    "llama-3.3-70b-specdec": 8192,
    # LLaMA 3.2
    "llama-3.2-90b-vision-preview": 8192,
    "llama-3.2-11b-vision-preview": 8192,
    "llama-3.2-3b-preview": 8192,
    "llama-3.2-1b-preview": 8192,
    # LLaMA 3.1
    "llama-3.1-70b-versatile": 128000,
    "llama-3.1-8b-instant": 128000,
    # LLaMA 3
    "llama3-70b-8192": 8192,
    "llama3-8b-8192": 8192,
    # LLaMA Guard
    "llama-guard-3-8b": 8192,
    # Mixtral
    "mixtral-8x7b-32768": 32768,
    # Gemma
    "gemma2-9b-it": 8192,
    "gemma-7b-it": 8192,
    # Whisper (audio)
    "whisper-large-v3": 25000,  # Audio tokens
    "whisper-large-v3-turbo": 25000,
    "distil-whisper-large-v3-en": 25000,
}

# Models with vision capability
VISION_MODELS = {
    "llama-3.2-90b-vision-preview",
    "llama-3.2-11b-vision-preview",
}

# Models supporting function calling (tool use)
FUNCTION_CALLING_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
}

# Rate limits (free tier)
# - 30 requests per minute
# - 6,000 tokens per minute
# - 14,400 requests per day


class GroqProvider(BaseLLMProvider):
    """
    Groq LLM Provider.

    Features:
    - Ultra-fast inference (~750 tokens/second)
    - Free tier available
    - LLaMA, Mixtral, Gemma models
    - Streaming support
    - Function calling (select models)
    - Vision support (LLaMA 3.2 vision models)

    Example:
        >>> config = LLMConfig(
        ...     provider="groq",
        ...     model="llama-3.3-70b-versatile",
        ...     api_key="gsk_..."
        ... )
        >>> async with GroqProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Groq provider.

        Args:
            config: LLM configuration. If not provided, uses defaults.
        """
        super().__init__(config)
        self._client: Any = None
        self._async_client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._api_key = config.api_key if config else None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model = self._model
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 8192)

        return ProviderInfo(
            name="groq",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Groq ultra-fast inference provider",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=model in FUNCTION_CALLING_MODELS,
                vision=model in VISION_MODELS,
                json_mode=True,  # Groq supports JSON mode
                embeddings=False,  # Groq doesn't have embedding API
                max_context_tokens=context_limit,
                max_output_tokens=min(8192, context_limit),
            ),
            supported_models=list(MODEL_CONTEXT_LIMITS.keys()),
            website="https://groq.com",
            documentation="https://console.groq.com/docs",
        )

    async def initialize(self) -> None:
        """Initialize the Groq client."""
        if self._initialized:
            return

        groq = _get_groq()

        # Resolve API key
        api_key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "groq",
                "API key not provided. Set GROQ_API_KEY environment variable or pass api_key in config."
            )

        # Create async client
        self._async_client = groq.AsyncGroq(
            api_key=api_key,
            timeout=self._config.timeout if self._config else 30,
        )
        self._client = groq.Groq(
            api_key=api_key,
            timeout=self._config.timeout if self._config else 30,
        )
        self._initialized = True

    async def close(self) -> None:
        """Close the Groq client."""
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
        Generate a completion using Groq's Chat API.

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
            GroqError: On API errors
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
        if "response_format" in kwargs:
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
                provider="groq",
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
        Stream a completion using Groq's Chat API.

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
            function_call: "auto", "none", or specific function
            **kwargs: Additional parameters

        Returns:
            LLMResponse with potential tool_calls
        """
        if self._model not in FUNCTION_CALLING_MODELS:
            raise InvalidModelError("groq", self._model)

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
                provider="groq",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                tool_calls=tool_calls,
                raw_response=response.model_dump() if hasattr(response, "model_dump") else None,
            )

        except Exception as e:
            self._handle_error(e)

    async def transcribe_audio(
        self,
        audio_file: Union[str, bytes],
        model: str = "whisper-large-v3",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Transcribe audio using Groq's Whisper models.

        Args:
            audio_file: Path to audio file or bytes
            model: Whisper model to use
            language: Language code (e.g., "en")
            prompt: Optional prompt to guide transcription
            response_format: "json", "text", "srt", "verbose_json", "vtt"
            temperature: Sampling temperature

        Returns:
            Transcription result

        Example:
            >>> result = await provider.transcribe_audio("audio.mp3")
            >>> print(result["text"])
        """
        self._ensure_initialized()

        # Handle file path vs bytes
        if isinstance(audio_file, str):
            with open(audio_file, "rb") as f:
                audio_data = f.read()
            filename = audio_file
        else:
            audio_data = audio_file
            filename = "audio.mp3"

        params: Dict[str, Any] = {
            "model": model,
            "file": (filename, audio_data),
            "response_format": response_format,
            "temperature": temperature,
        }

        if language:
            params["language"] = language
        if prompt:
            params["prompt"] = prompt

        try:
            response = await self._async_client.audio.transcriptions.create(**params)

            if response_format == "json" or response_format == "verbose_json":
                return {"text": response.text}
            return {"text": response}

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """
        Handle Groq API errors and convert to Sentimatrix exceptions.

        Args:
            error: Exception to handle

        Raises:
            Appropriate Sentimatrix exception
        """
        groq = _get_groq()

        error_message = str(error)

        # Handle Groq-specific errors
        if isinstance(error, groq.AuthenticationError):
            raise AuthenticationError("groq", "Invalid API key") from error

        if isinstance(error, groq.RateLimitError):
            raise RateLimitError(
                "Rate limit exceeded. Groq free tier: 30 req/min, 6000 tokens/min",
                provider="groq",
                retry_after=60,
            ) from error

        if isinstance(error, groq.BadRequestError):
            if "context_length" in error_message.lower():
                raise TokenLimitExceededError(
                    "groq",
                    self._model,
                    requested=0,
                    limit=MODEL_CONTEXT_LIMITS.get(self._model, 0),
                ) from error
            if "content" in error_message.lower() and "filter" in error_message.lower():
                raise ContentFilteredError("groq") from error
            raise GroqError(error_message, self._model, original_error=error)

        if isinstance(error, groq.NotFoundError):
            raise InvalidModelError("groq", self._model) from error

        if isinstance(error, groq.APITimeoutError):
            from sentimatrix.core.exceptions import TimeoutError
            raise TimeoutError(
                f"Groq API timeout: {error_message}",
                timeout=self._config.timeout if self._config else 30,
                operation="generate",
            ) from error

        if isinstance(error, groq.APIConnectionError):
            from sentimatrix.core.exceptions import ConnectionTimeoutError
            raise ConnectionTimeoutError(
                "api.groq.com",
                self._config.timeout if self._config else 30,
            ) from error

        # Generic Groq error
        if hasattr(groq, 'GroqError') and isinstance(error, groq.GroqError):
            raise GroqError(error_message, self._model, original_error=error)

        # Re-raise unknown errors
        raise GroqError(
            f"Unexpected error: {error_message}",
            self._model,
            original_error=error,
        )


# Register the provider
register_provider("groq", ProviderType.LLM, GroqProvider)
