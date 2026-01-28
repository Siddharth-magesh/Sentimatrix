"""
Google Gemini LLM Provider

Implements the BaseLLMProvider interface for Google's Gemini API.
Supports Gemini 2.0, 1.5 Pro/Flash, and embedding models.

Official API Documentation: https://ai.google.dev/gemini-api/docs
Free Tier: Available (60 requests/minute for Gemini 1.5 Flash)
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
    GeminiError,
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

# Lazy imports
_genai = None


def _get_genai():
    """Lazy import of google.generativeai module."""
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai
            _genai = genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Gemini provider. "
                "Install it with: pip install google-generativeai"
            )
    return _genai


# Default model
DEFAULT_MODEL = "gemini-1.5-flash"

# Available models and their context limits
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # Gemini 2.0
    "gemini-2.0-flash-exp": 1048576,  # 1M tokens
    "gemini-2.0-flash-thinking-exp": 32767,
    # Gemini 1.5
    "gemini-1.5-pro": 2097152,  # 2M tokens
    "gemini-1.5-pro-latest": 2097152,
    "gemini-1.5-flash": 1048576,  # 1M tokens
    "gemini-1.5-flash-latest": 1048576,
    "gemini-1.5-flash-8b": 1048576,
    # Gemini 1.0
    "gemini-1.0-pro": 32760,
    "gemini-pro": 32760,
    # Vision models (also support text)
    "gemini-1.5-pro-vision": 2097152,
    "gemini-pro-vision": 16384,
    # Embedding models
    "text-embedding-004": 2048,
    "embedding-001": 2048,
}

# Max output tokens
MODEL_MAX_OUTPUT: Dict[str, int] = {
    "gemini-2.0-flash-exp": 8192,
    "gemini-2.0-flash-thinking-exp": 8192,
    "gemini-1.5-pro": 8192,
    "gemini-1.5-pro-latest": 8192,
    "gemini-1.5-flash": 8192,
    "gemini-1.5-flash-latest": 8192,
    "gemini-1.5-flash-8b": 8192,
    "gemini-1.0-pro": 8192,
    "gemini-pro": 8192,
}

# Models with vision capability
VISION_MODELS = {
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro", "gemini-1.5-pro-latest", "gemini-1.5-pro-vision",
    "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-8b",
    "gemini-pro-vision",
}

# Models with function calling
FUNCTION_CALLING_MODELS = {
    "gemini-2.0-flash-exp",
    "gemini-1.5-pro", "gemini-1.5-pro-latest",
    "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash-8b",
    "gemini-1.0-pro", "gemini-pro",
}

# Embedding models
EMBEDDING_MODELS = {"text-embedding-004", "embedding-001"}


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini LLM Provider.

    Features:
    - Gemini 2.0/1.5 Pro/Flash models
    - Up to 2M token context (Gemini 1.5 Pro)
    - Streaming support
    - Function calling
    - Vision support (multimodal)
    - Text embeddings
    - Free tier available

    Example:
        >>> config = LLMConfig(
        ...     provider="gemini",
        ...     model="gemini-1.5-flash",
        ...     api_key="..."
        ... )
        >>> async with GeminiProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Gemini provider.

        Args:
            config: LLM configuration. If not provided, uses defaults.
        """
        super().__init__(config)
        self._model_instance: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._api_key = config.api_key if config else None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model = self._model
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 32760)
        max_output = MODEL_MAX_OUTPUT.get(model, 8192)

        return ProviderInfo(
            name="gemini",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Google Gemini models provider",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=model in FUNCTION_CALLING_MODELS,
                vision=model in VISION_MODELS,
                json_mode=True,  # Via response_mime_type
                embeddings=model in EMBEDDING_MODELS or model not in EMBEDDING_MODELS,  # All models can use embedding endpoint
                max_context_tokens=context_limit,
                max_output_tokens=max_output,
            ),
            supported_models=list(MODEL_CONTEXT_LIMITS.keys()),
            website="https://ai.google.dev",
            documentation="https://ai.google.dev/gemini-api/docs",
        )

    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        if self._initialized:
            return

        genai = _get_genai()

        # Resolve API key
        api_key = self._api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "gemini",
                "API key not provided. Set GOOGLE_API_KEY environment variable or pass api_key in config."
            )

        # Configure the API
        genai.configure(api_key=api_key)

        # Create model instance
        self._model_instance = genai.GenerativeModel(self._model)
        self._initialized = True

    async def close(self) -> None:
        """Close the Gemini client (no-op, stateless)."""
        self._model_instance = None
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
        Generate a completion using Gemini's API.

        Args:
            prompt: User message
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters (response_mime_type, etc.)

        Returns:
            LLMResponse with generated content

        Raises:
            GeminiError: On API errors
        """
        self._ensure_initialized()

        genai = _get_genai()
        start_time = time.perf_counter()

        # Build generation config
        generation_config: Dict[str, Any] = {}

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        generation_config["temperature"] = temp

        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )
        generation_config["max_output_tokens"] = max_tok

        if stop:
            generation_config["stop_sequences"] = stop

        # Handle JSON mode
        if kwargs.get("response_format", {}).get("type") == "json_object":
            generation_config["response_mime_type"] = "application/json"
        elif "response_mime_type" in kwargs:
            generation_config["response_mime_type"] = kwargs.pop("response_mime_type")

        # Handle additional config
        if "top_p" in kwargs:
            generation_config["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            generation_config["top_k"] = kwargs.pop("top_k")

        # Create model with system instruction if provided
        if system_prompt:
            model = genai.GenerativeModel(
                self._model,
                system_instruction=system_prompt,
            )
        else:
            model = self._model_instance

        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(**generation_config),
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract content
            content = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            content += part.text

            # Determine finish reason
            finish_reason = "stop"
            if response.candidates:
                reason = response.candidates[0].finish_reason
                if reason:
                    # Convert Gemini finish reason to standard
                    reason_map = {
                        1: "stop",  # STOP
                        2: "length",  # MAX_TOKENS
                        3: "content_filter",  # SAFETY
                        4: "content_filter",  # RECITATION
                        5: "other",  # OTHER
                    }
                    finish_reason = reason_map.get(reason.value if hasattr(reason, 'value') else reason, "stop")

            # Build usage stats
            usage = TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                completion_tokens=response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                total_tokens=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
            )

            return LLMResponse(
                content=content,
                model=self._model,
                provider="gemini",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                raw_response=None,  # Gemini response is not easily serializable
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
        Stream a completion using Gemini's API.

        Args:
            prompt: User message
            system_prompt: Optional system instruction
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters

        Yields:
            Text chunks as they're generated
        """
        self._ensure_initialized()

        genai = _get_genai()

        # Build generation config
        generation_config: Dict[str, Any] = {}

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        generation_config["temperature"] = temp

        max_tok = max_tokens if max_tokens is not None else (
            self._config.max_tokens if self._config else 1024
        )
        generation_config["max_output_tokens"] = max_tok

        if stop:
            generation_config["stop_sequences"] = stop

        # Create model with system instruction if provided
        if system_prompt:
            model = genai.GenerativeModel(
                self._model,
                system_instruction=system_prompt,
            )
        else:
            model = self._model_instance

        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(**generation_config),
                stream=True,
            )

            async for chunk in response:
                if chunk.text:
                    yield chunk.text

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
            functions: List of function definitions
            system_prompt: Optional system instruction
            function_call: "auto", "none", or specific function
            **kwargs: Additional parameters

        Returns:
            LLMResponse with potential tool_calls

        Note:
            Gemini uses a specific tool format. This method accepts
            OpenAI-style tools and converts them.
        """
        if self._model not in FUNCTION_CALLING_MODELS:
            raise InvalidModelError("gemini", self._model)

        self._ensure_initialized()

        genai = _get_genai()
        start_time = time.perf_counter()

        # Convert functions to Gemini tool format
        tools = []
        for func in functions:
            if "type" in func and func["type"] == "function":
                # OpenAI format
                tool_def = {
                    "name": func["function"]["name"],
                    "description": func["function"].get("description", ""),
                    "parameters": func["function"].get("parameters", {"type": "object", "properties": {}}),
                }
            else:
                # Direct format
                tool_def = func
            tools.append(tool_def)

        # Build generation config
        generation_config: Dict[str, Any] = {}

        temp = kwargs.get("temperature", self._config.temperature if self._config else 0.7)
        generation_config["temperature"] = temp

        max_tok = kwargs.get("max_tokens", self._config.max_tokens if self._config else 1024)
        generation_config["max_output_tokens"] = max_tok

        # Create model with tools
        if system_prompt:
            model = genai.GenerativeModel(
                self._model,
                system_instruction=system_prompt,
                tools=tools,
            )
        else:
            model = genai.GenerativeModel(
                self._model,
                tools=tools,
            )

        try:
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.GenerationConfig(**generation_config),
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract content and function calls
            content = ""
            tool_calls = []

            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for i, part in enumerate(candidate.content.parts):
                        if hasattr(part, "text") and part.text:
                            content += part.text
                        elif hasattr(part, "function_call"):
                            fc = part.function_call
                            tool_calls.append({
                                "id": f"call_{i}",
                                "type": "function",
                                "function": {
                                    "name": fc.name,
                                    "arguments": str(dict(fc.args)) if fc.args else "{}",
                                }
                            })

            finish_reason = "stop"
            if tool_calls:
                finish_reason = "tool_calls"
            elif response.candidates:
                reason = response.candidates[0].finish_reason
                if reason:
                    reason_map = {1: "stop", 2: "length", 3: "content_filter", 4: "content_filter", 5: "other"}
                    finish_reason = reason_map.get(reason.value if hasattr(reason, 'value') else reason, "stop")

            usage = TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                completion_tokens=response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                total_tokens=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
            )

            return LLMResponse(
                content=content,
                model=self._model,
                provider="gemini",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                tool_calls=tool_calls if tool_calls else None,
                raw_response=None,
            )

        except Exception as e:
            self._handle_error(e)

    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion with image inputs.

        Args:
            prompt: User message
            images: List of image URLs, file paths, or bytes
            system_prompt: Optional system instruction
            **kwargs: Additional parameters

        Returns:
            LLMResponse with generated content
        """
        if self._model not in VISION_MODELS:
            raise InvalidModelError("gemini", self._model)

        self._ensure_initialized()

        genai = _get_genai()
        import base64

        start_time = time.perf_counter()

        # Build content parts
        content_parts = []

        for image in images:
            if isinstance(image, bytes):
                # Raw bytes
                content_parts.append({
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image).decode("utf-8"),
                })
            elif image.startswith("data:"):
                # Data URL
                parts = image.split(",", 1)
                if len(parts) == 2:
                    header, data = parts
                    mime_type = header.split(";")[0].replace("data:", "")
                    content_parts.append({
                        "mime_type": mime_type,
                        "data": data,
                    })
            elif image.startswith("http"):
                # URL - need to download first for Gemini
                # For simplicity, we'll use the URL directly if supported
                import httpx
                async with httpx.AsyncClient() as client:
                    resp = await client.get(image)
                    content_parts.append({
                        "mime_type": resp.headers.get("content-type", "image/jpeg"),
                        "data": base64.b64encode(resp.content).decode("utf-8"),
                    })
            else:
                # Assume file path
                with open(image, "rb") as f:
                    data = f.read()
                # Detect mime type from extension
                ext = image.lower().split(".")[-1]
                mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}
                content_parts.append({
                    "mime_type": mime_map.get(ext, "image/jpeg"),
                    "data": base64.b64encode(data).decode("utf-8"),
                })

        # Add text prompt
        content_parts.append(prompt)

        # Build generation config
        generation_config: Dict[str, Any] = {}
        temp = kwargs.get("temperature", self._config.temperature if self._config else 0.7)
        generation_config["temperature"] = temp
        max_tok = kwargs.get("max_tokens", self._config.max_tokens if self._config else 1024)
        generation_config["max_output_tokens"] = max_tok

        # Create model with system instruction if provided
        if system_prompt:
            model = genai.GenerativeModel(
                self._model,
                system_instruction=system_prompt,
            )
        else:
            model = self._model_instance

        try:
            response = await model.generate_content_async(
                content_parts,
                generation_config=genai.GenerationConfig(**generation_config),
            )

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            content = ""
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            content += part.text

            finish_reason = "stop"
            if response.candidates:
                reason = response.candidates[0].finish_reason
                if reason:
                    reason_map = {1: "stop", 2: "length", 3: "content_filter", 4: "content_filter", 5: "other"}
                    finish_reason = reason_map.get(reason.value if hasattr(reason, 'value') else reason, "stop")

            usage = TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                completion_tokens=response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
                total_tokens=response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') and response.usage_metadata else 0,
            )

            return LLMResponse(
                content=content,
                model=self._model,
                provider="gemini",
                usage=usage,
                finish_reason=finish_reason,
                response_time_ms=elapsed_ms,
                raw_response=None,
            )

        except Exception as e:
            self._handle_error(e)

    async def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using Gemini's embedding API.

        Args:
            text: Single text or list of texts
            model: Embedding model (default: text-embedding-004)

        Returns:
            Embedding vector(s)
        """
        self._ensure_initialized()

        genai = _get_genai()

        embedding_model = model or "text-embedding-004"

        # Handle single text vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        try:
            result = genai.embed_content(
                model=f"models/{embedding_model}",
                content=texts,
                task_type="retrieval_document",
            )

            embeddings = result["embedding"]

            # Handle single vs batch
            if is_single:
                return embeddings if isinstance(embeddings[0], float) else embeddings[0]
            return embeddings

        except Exception as e:
            self._handle_error(e)

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens using Gemini's tokenizer.

        Args:
            text: Text to count tokens
            model: Model (unused, uses configured model)

        Returns:
            Number of tokens
        """
        if not self._initialized or not self._model_instance:
            # Rough estimate if not initialized
            return len(text) // 4

        try:
            result = self._model_instance.count_tokens(text)
            return result.total_tokens
        except Exception:
            # Fallback to estimate
            return len(text) // 4

    def _handle_error(self, error: Exception) -> None:
        """
        Handle Gemini API errors and convert to Sentimatrix exceptions.

        Args:
            error: Exception to handle

        Raises:
            Appropriate Sentimatrix exception
        """
        error_message = str(error)
        error_type = type(error).__name__

        # Check for specific error types
        if "InvalidArgument" in error_type or "invalid" in error_message.lower():
            if "api_key" in error_message.lower() or "api key" in error_message.lower():
                raise AuthenticationError("gemini", "Invalid API key") from error
            if "model" in error_message.lower():
                raise InvalidModelError("gemini", self._model) from error

        if "ResourceExhausted" in error_type or "quota" in error_message.lower() or "rate" in error_message.lower():
            raise RateLimitError(
                "Rate limit exceeded",
                provider="gemini",
                retry_after=60,
            ) from error

        if "PermissionDenied" in error_type:
            raise AuthenticationError("gemini", "Permission denied") from error

        if "blocked" in error_message.lower() or "safety" in error_message.lower():
            raise ContentFilteredError("gemini") from error

        if "token" in error_message.lower() and ("limit" in error_message.lower() or "exceed" in error_message.lower()):
            raise TokenLimitExceededError(
                "gemini",
                self._model,
                requested=0,
                limit=MODEL_CONTEXT_LIMITS.get(self._model, 0),
            ) from error

        if "Timeout" in error_type or "timeout" in error_message.lower():
            from sentimatrix.core.exceptions import TimeoutError
            raise TimeoutError(
                f"Gemini API timeout: {error_message}",
                timeout=self._config.timeout if self._config else 30,
                operation="generate",
            ) from error

        # Generic error
        raise GeminiError(
            f"Gemini error: {error_message}",
            self._model,
            original_error=error,
        )


# Register the provider
register_provider("gemini", ProviderType.LLM, GeminiProvider)
