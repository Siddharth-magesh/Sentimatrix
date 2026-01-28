"""
Ollama LLM Provider

Implements the BaseLLMProvider interface for Ollama local models.
Supports running models locally without requiring API keys.

Official Documentation: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    ErrorCode,
    InvalidModelError,
    OllamaError,
    ProviderInitializationError,
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
                "httpx package is required for Ollama provider. "
                "Install it with: pip install httpx"
            )
    return _httpx


# Default configuration
DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

# Common models and their approximate context limits
# These are approximate and depend on the actual model configuration
MODEL_CONTEXT_LIMITS: Dict[str, int] = {
    # LLaMA 3.2
    "llama3.2": 128000,
    "llama3.2:1b": 128000,
    "llama3.2:3b": 128000,
    # LLaMA 3.1
    "llama3.1": 128000,
    "llama3.1:8b": 128000,
    "llama3.1:70b": 128000,
    "llama3.1:405b": 128000,
    # LLaMA 3
    "llama3": 8192,
    "llama3:8b": 8192,
    "llama3:70b": 8192,
    # LLaMA 2
    "llama2": 4096,
    "llama2:7b": 4096,
    "llama2:13b": 4096,
    "llama2:70b": 4096,
    # Mistral
    "mistral": 32768,
    "mistral:7b": 32768,
    "mistral-nemo": 128000,
    # Mixtral
    "mixtral": 32768,
    "mixtral:8x7b": 32768,
    "mixtral:8x22b": 65536,
    # Phi
    "phi3": 128000,
    "phi3:mini": 128000,
    "phi3:medium": 128000,
    # Gemma
    "gemma": 8192,
    "gemma:2b": 8192,
    "gemma:7b": 8192,
    "gemma2": 8192,
    "gemma2:2b": 8192,
    "gemma2:9b": 8192,
    "gemma2:27b": 8192,
    # Qwen
    "qwen": 32768,
    "qwen:7b": 32768,
    "qwen:14b": 32768,
    "qwen:72b": 32768,
    "qwen2": 128000,
    "qwen2.5": 128000,
    # CodeLlama
    "codellama": 16384,
    "codellama:7b": 16384,
    "codellama:13b": 16384,
    "codellama:34b": 16384,
    # DeepSeek
    "deepseek-coder": 16384,
    "deepseek-coder-v2": 128000,
    # Starcoder
    "starcoder": 8192,
    "starcoder2": 16384,
    # LLaVA (vision)
    "llava": 4096,
    "llava:7b": 4096,
    "llava:13b": 4096,
    "llava:34b": 4096,
    # Vicuna
    "vicuna": 2048,
    # Neural Chat
    "neural-chat": 8192,
    # Orca
    "orca-mini": 2048,
    "orca2": 4096,
    # Falcon
    "falcon": 2048,
    "falcon:7b": 2048,
    "falcon:40b": 2048,
    # Yi
    "yi": 4096,
    "yi:6b": 4096,
    "yi:34b": 4096,
    # Dolphin
    "dolphin-mixtral": 32768,
    "dolphin-llama3": 8192,
    # Nomic
    "nomic-embed-text": 8192,
}

# Models with vision capability
VISION_MODELS = {"llava", "llava:7b", "llava:13b", "llava:34b", "llama3.2-vision", "moondream"}

# Embedding models
EMBEDDING_MODELS = {"nomic-embed-text", "mxbai-embed-large", "all-minilm"}


class OllamaProvider(BaseLLMProvider):
    """
    Ollama Local LLM Provider.

    Features:
    - Local inference (no API key required)
    - Wide model support (LLaMA, Mistral, Mixtral, etc.)
    - Streaming support
    - Vision support (LLaVA)
    - Embeddings support
    - Pull models on demand

    Example:
        >>> config = LLMConfig(
        ...     provider="ollama",
        ...     model="llama3.2",
        ...     api_base="http://localhost:11434"
        ... )
        >>> async with OllamaProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)

    Note:
        Requires Ollama to be running locally.
        Install from: https://ollama.ai
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration. If not provided, uses defaults.
        """
        super().__init__(config)
        self._client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._host = config.api_base if config and config.api_base else DEFAULT_HOST
        self._available_models: List[str] = []

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model = self._model
        # Try to get context limit, default to 4096 for unknown models
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 4096)

        return ProviderInfo(
            name="ollama",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Ollama local models provider",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=False,  # Ollama has limited function calling
                vision=model in VISION_MODELS or "vision" in model.lower(),
                json_mode=True,  # Via format parameter
                embeddings=True,
                max_context_tokens=context_limit,
                max_output_tokens=min(4096, context_limit),
            ),
            supported_models=list(MODEL_CONTEXT_LIMITS.keys()),
            website="https://ollama.ai",
            documentation="https://github.com/ollama/ollama/blob/main/docs/api.md",
        )

    async def initialize(self) -> None:
        """Initialize the Ollama client and verify connection."""
        if self._initialized:
            return

        httpx = _get_httpx()

        # Create async client
        self._client = httpx.AsyncClient(
            base_url=self._host,
            timeout=self._config.timeout if self._config else 120,
        )

        # Verify Ollama is running
        try:
            response = await self._client.get("/api/tags")
            if response.status_code == 200:
                data = response.json()
                self._available_models = [m["name"] for m in data.get("models", [])]
            else:
                raise ProviderInitializationError(
                    "ollama",
                    f"Failed to connect to Ollama at {self._host}. "
                    f"Status: {response.status_code}"
                )
        except Exception as e:
            if "ConnectError" in str(type(e).__name__) or "Connection" in str(e):
                raise ProviderInitializationError(
                    "ollama",
                    f"Cannot connect to Ollama at {self._host}. "
                    "Make sure Ollama is running: ollama serve"
                )
            raise ProviderInitializationError("ollama", str(e))

        self._initialized = True

    async def close(self) -> None:
        """Close the HTTP client."""
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
        Generate a completion using Ollama's API.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (num_predict)
            stop: Stop sequences
            **kwargs: Additional options (format, etc.)

        Returns:
            LLMResponse with generated content

        Raises:
            OllamaError: On API errors
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        # Build request body
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
        }

        if system_prompt:
            body["system"] = system_prompt

        # Build options
        options: Dict[str, Any] = {}

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        options["temperature"] = temp

        if max_tokens is not None:
            options["num_predict"] = max_tokens
        elif self._config and self._config.max_tokens:
            options["num_predict"] = self._config.max_tokens

        if stop:
            options["stop"] = stop

        # Handle additional options
        if "top_p" in kwargs:
            options["top_p"] = kwargs.pop("top_p")
        if "top_k" in kwargs:
            options["top_k"] = kwargs.pop("top_k")
        if "repeat_penalty" in kwargs:
            options["repeat_penalty"] = kwargs.pop("repeat_penalty")
        if "seed" in kwargs:
            options["seed"] = kwargs.pop("seed")

        if options:
            body["options"] = options

        # Handle JSON format
        if kwargs.get("format") == "json":
            body["format"] = "json"
        elif kwargs.get("response_format", {}).get("type") == "json_object":
            body["format"] = "json"

        try:
            response = await self._client.post("/api/generate", json=body)

            if response.status_code != 200:
                error_text = response.text
                raise OllamaError(
                    f"Ollama API error: {error_text}",
                    self._model,
                )

            data = response.json()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            content = data.get("response", "")
            done_reason = data.get("done_reason", "stop")

            # Build usage stats (Ollama provides token counts)
            usage = TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=(
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            )

            return LLMResponse(
                content=content,
                model=data.get("model", self._model),
                provider="ollama",
                usage=usage,
                finish_reason=done_reason,
                response_time_ms=elapsed_ms,
                raw_response=data,
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
        Stream a completion using Ollama's API.

        Args:
            prompt: User message
            system_prompt: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional options

        Yields:
            Text chunks as they're generated
        """
        self._ensure_initialized()

        # Build request body
        body: Dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": True,
        }

        if system_prompt:
            body["system"] = system_prompt

        options: Dict[str, Any] = {}

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        options["temperature"] = temp

        if max_tokens is not None:
            options["num_predict"] = max_tokens
        elif self._config and self._config.max_tokens:
            options["num_predict"] = self._config.max_tokens

        if stop:
            options["stop"] = stop

        if options:
            body["options"] = options

        try:
            async with self._client.stream("POST", "/api/generate", json=body) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise OllamaError(
                        f"Ollama API error: {error_text.decode()}",
                        self._model,
                    )

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            self._handle_error(e)

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate using Ollama's chat API (multi-turn conversation).

        Args:
            messages: List of message dicts with "role" and "content"
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional options

        Returns:
            LLMResponse with generated content

        Example:
            >>> messages = [
            ...     {"role": "user", "content": "Hello!"},
            ...     {"role": "assistant", "content": "Hi there!"},
            ...     {"role": "user", "content": "How are you?"}
            ... ]
            >>> response = await provider.chat(messages)
        """
        self._ensure_initialized()

        start_time = time.perf_counter()

        body: Dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        options: Dict[str, Any] = {}

        temp = temperature if temperature is not None else (
            self._config.temperature if self._config else 0.7
        )
        options["temperature"] = temp

        if max_tokens is not None:
            options["num_predict"] = max_tokens

        if stop:
            options["stop"] = stop

        if options:
            body["options"] = options

        if kwargs.get("format") == "json":
            body["format"] = "json"

        try:
            response = await self._client.post("/api/chat", json=body)

            if response.status_code != 200:
                error_text = response.text
                raise OllamaError(f"Ollama chat API error: {error_text}", self._model)

            data = response.json()
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            message = data.get("message", {})
            content = message.get("content", "")
            done_reason = data.get("done_reason", "stop")

            usage = TokenUsage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=(
                    data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
                ),
            )

            return LLMResponse(
                content=content,
                model=data.get("model", self._model),
                provider="ollama",
                usage=usage,
                finish_reason=done_reason,
                response_time_ms=elapsed_ms,
                raw_response=data,
            )

        except Exception as e:
            self._handle_error(e)

    async def embed(
        self,
        text: Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings using Ollama's embedding API.

        Args:
            text: Single text or list of texts
            model: Embedding model (default: nomic-embed-text)

        Returns:
            Embedding vector(s)

        Example:
            >>> embedding = await provider.embed("Hello world")
            >>> embeddings = await provider.embed(["Hello", "World"])
        """
        self._ensure_initialized()

        embedding_model = model or "nomic-embed-text"

        # Handle single text vs list
        is_single = isinstance(text, str)
        texts = [text] if is_single else text

        embeddings = []

        for t in texts:
            body = {
                "model": embedding_model,
                "prompt": t,
            }

            try:
                response = await self._client.post("/api/embeddings", json=body)

                if response.status_code != 200:
                    raise OllamaError(
                        f"Ollama embeddings error: {response.text}",
                        embedding_model,
                    )

                data = response.json()
                embeddings.append(data.get("embedding", []))

            except Exception as e:
                self._handle_error(e)

        return embeddings[0] if is_single else embeddings

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models in Ollama.

        Returns:
            List of model information dictionaries

        Example:
            >>> models = await provider.list_models()
            >>> for model in models:
            ...     print(model["name"])
        """
        self._ensure_initialized()

        try:
            response = await self._client.get("/api/tags")

            if response.status_code != 200:
                raise OllamaError(
                    f"Failed to list models: {response.text}",
                    self._model,
                )

            data = response.json()
            return data.get("models", [])

        except Exception as e:
            self._handle_error(e)

    async def pull_model(self, model_name: str) -> AsyncIterator[Dict[str, Any]]:
        """
        Pull/download a model from Ollama library.

        Args:
            model_name: Name of the model to pull

        Yields:
            Progress updates

        Example:
            >>> async for progress in provider.pull_model("llama3.2"):
            ...     print(f"Progress: {progress.get('completed', 0)}/{progress.get('total', 0)}")
        """
        self._ensure_initialized()

        body = {"name": model_name, "stream": True}

        try:
            async with self._client.stream("POST", "/api/pull", json=body) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    raise OllamaError(
                        f"Failed to pull model: {error_text.decode()}",
                        model_name,
                    )

                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            continue

        except Exception as e:
            self._handle_error(e)

    async def model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.

        Args:
            model_name: Model name (defaults to configured model)

        Returns:
            Model information dictionary
        """
        self._ensure_initialized()

        model = model_name or self._model
        body = {"name": model}

        try:
            response = await self._client.post("/api/show", json=body)

            if response.status_code != 200:
                raise OllamaError(
                    f"Failed to get model info: {response.text}",
                    model,
                )

            return response.json()

        except Exception as e:
            self._handle_error(e)

    def _handle_error(self, error: Exception) -> None:
        """
        Handle Ollama errors and convert to Sentimatrix exceptions.

        Args:
            error: Exception to handle

        Raises:
            Appropriate Sentimatrix exception
        """
        error_message = str(error)

        # Already a Sentimatrix error
        if isinstance(error, OllamaError):
            raise error

        # Connection errors
        if "ConnectError" in str(type(error).__name__) or "Connection" in error_message:
            raise OllamaError(
                f"Cannot connect to Ollama at {self._host}. "
                "Make sure Ollama is running: ollama serve",
                self._model,
                original_error=error,
            )

        # Timeout
        if "TimeoutError" in str(type(error).__name__) or "timeout" in error_message.lower():
            from sentimatrix.core.exceptions import TimeoutError
            raise TimeoutError(
                f"Ollama request timeout: {error_message}",
                timeout=self._config.timeout if self._config else 120,
                operation="generate",
            )

        # Model not found
        if "not found" in error_message.lower() or "pull" in error_message.lower():
            raise InvalidModelError("ollama", self._model)

        # Generic error
        raise OllamaError(
            f"Ollama error: {error_message}",
            self._model,
            original_error=error,
        )


# Register the provider
register_provider("ollama", ProviderType.LLM, OllamaProvider)
