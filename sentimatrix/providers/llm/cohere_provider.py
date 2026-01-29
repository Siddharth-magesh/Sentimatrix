"""
Cohere LLM Provider

Implements the BaseLLMProvider interface for Cohere.
Enterprise AI platform specializing in RAG and embeddings.

Official API Documentation: https://docs.cohere.com/
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
_cohere = None
_httpx = None


def _get_cohere():
    """Lazy import of cohere module."""
    global _cohere
    if _cohere is None:
        try:
            import cohere
            _cohere = cohere
        except ImportError:
            raise ImportError(
                "cohere package is required for Cohere provider. "
                "Install it with: pip install cohere"
            )
    return _cohere


def _get_httpx():
    """Lazy import of httpx for fallback."""
    global _httpx
    if _httpx is None:
        try:
            import httpx
            _httpx = httpx
        except ImportError:
            pass
    return _httpx


# Default model
DEFAULT_MODEL = "command-r-plus"

# Model configurations
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "command-r-plus": {
        "context": 128000,
        "description": "Most powerful Command model"
    },
    "command-r": {
        "context": 128000,
        "description": "Balanced performance and efficiency"
    },
    "command": {
        "context": 4096,
        "description": "Legacy command model"
    },
    "command-light": {
        "context": 4096,
        "description": "Faster, lighter version"
    },
    "command-nightly": {
        "context": 128000,
        "description": "Latest experimental features"
    },
    # Embedding models
    "embed-english-v3.0": {
        "context": 512,
        "description": "English embeddings"
    },
    "embed-multilingual-v3.0": {
        "context": 512,
        "description": "Multilingual embeddings"
    },
    "embed-english-light-v3.0": {
        "context": 512,
        "description": "Lightweight English embeddings"
    },
    # Rerank models
    "rerank-english-v3.0": {
        "context": 4096,
        "description": "English reranking"
    },
    "rerank-multilingual-v3.0": {
        "context": 4096,
        "description": "Multilingual reranking"
    },
}


class CohereProvider(BaseLLMProvider):
    """
    Cohere LLM Provider.

    Enterprise AI platform with specialized capabilities:
    - Command R+ (advanced reasoning)
    - RAG (Retrieval-Augmented Generation)
    - Best-in-class embeddings
    - Document reranking
    - Multi-language support

    Supports:
    - Chat completions
    - Streaming responses
    - Text embeddings
    - Document reranking
    - RAG with citations

    Example:
        >>> config = LLMConfig(
        ...     provider="cohere",
        ...     api_key="your-api-key",
        ...     model="command-r-plus",
        ... )
        >>> async with CohereProvider(config) as provider:
        ...     response = await provider.generate("Hello!")
        ...     print(response.content)
    """

    API_BASE = "https://api.cohere.ai/v1"

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize Cohere provider.

        Args:
            config: LLM configuration with Cohere API key.
        """
        super().__init__(config)
        self._client: Any = None
        self._async_client: Any = None
        self._model = config.model if config else DEFAULT_MODEL
        self._api_key = config.api_key if config else None

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        model_config = MODEL_CONFIGS.get(self._model, {})
        context_limit = model_config.get("context", 128000)

        return ProviderInfo(
            name="cohere",
            provider_type=ProviderType.LLM,
            version="1.0.0",
            description="Cohere - Enterprise AI with RAG and embeddings",
            capabilities=ProviderCapabilities(
                streaming=True,
                function_calling=True,
                vision=False,
                json_mode=True,
                embeddings=True,
                max_context_tokens=context_limit,
                max_output_tokens=4096,
            ),
            supported_models=list(MODEL_CONFIGS.keys()),
            website="https://cohere.com",
            documentation="https://docs.cohere.com/",
        )

    async def initialize(self) -> None:
        """Initialize the Cohere client."""
        if self._initialized:
            return

        api_key = self._api_key or os.environ.get("COHERE_API_KEY")
        if not api_key:
            raise AuthenticationError(
                "cohere",
                "API key not provided. Set COHERE_API_KEY environment variable "
                "or pass api_key in config."
            )

        try:
            cohere = _get_cohere()
            self._client = cohere.Client(api_key=api_key)
            self._async_client = cohere.AsyncClient(api_key=api_key)
            self._initialized = True
        except ImportError:
            # Fallback to httpx
            httpx = _get_httpx()
            if httpx:
                self._async_client = httpx.AsyncClient(
                    base_url=self.API_BASE,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self._config.timeout if self._config else 60,
                )
                self._initialized = True
            else:
                raise ImportError(
                    "Either cohere or httpx package is required. "
                    "Install with: pip install cohere"
                )

    async def close(self) -> None:
        """Close the Cohere client."""
        if hasattr(self._async_client, 'aclose'):
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
        Generate a completion using Cohere.

        Args:
            prompt: User message
            system_prompt: Optional system/preamble message
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional parameters (documents, connectors, etc.)

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

        try:
            cohere = _get_cohere()

            # Use v2 chat API
            params = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": max_tok,
            }

            if system_prompt:
                params["preamble"] = system_prompt

            if stop:
                params["stop_sequences"] = stop

            # RAG-specific options
            if "documents" in kwargs:
                params["documents"] = kwargs["documents"]

            if "connectors" in kwargs:
                params["connectors"] = kwargs["connectors"]

            if "citation_quality" in kwargs:
                params["citation_quality"] = kwargs["citation_quality"]

            response = await self._async_client.chat(**params)

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract usage
            usage = TokenUsage(
                prompt_tokens=getattr(response.meta, 'billed_units', {}).get('input_tokens', 0) if hasattr(response, 'meta') else 0,
                completion_tokens=getattr(response.meta, 'billed_units', {}).get('output_tokens', 0) if hasattr(response, 'meta') else 0,
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

            return LLMResponse(
                content=response.text,
                model=self._model,
                provider="cohere",
                usage=usage,
                finish_reason=response.finish_reason if hasattr(response, 'finish_reason') else "stop",
                response_time_ms=elapsed_ms,
                raw_response=response,
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
        """Stream a completion using Cohere."""
        self._ensure_initialized()

        temp = temperature if temperature is not None else 0.7
        max_tok = max_tokens if max_tokens is not None else 1024

        try:
            cohere = _get_cohere()

            params = {
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temp,
                "max_tokens": max_tok,
            }

            if system_prompt:
                params["preamble"] = system_prompt

            if stop:
                params["stop_sequences"] = stop

            async for event in self._async_client.chat_stream(**params):
                if hasattr(event, 'text'):
                    yield event.text

        except Exception as e:
            raise ProviderError("cohere", str(e)) from e

    async def embed(
        self,
        texts: List[str],
        model: Optional[str] = None,
        input_type: str = "search_document",
    ) -> List[List[float]]:
        """
        Generate embeddings using Cohere.

        Args:
            texts: List of texts to embed
            model: Embedding model (default: embed-english-v3.0)
            input_type: Type of text (search_document, search_query, classification, clustering)

        Returns:
            List of embedding vectors
        """
        self._ensure_initialized()

        embed_model = model or "embed-english-v3.0"

        try:
            response = await self._async_client.embed(
                texts=texts,
                model=embed_model,
                input_type=input_type,
            )

            return response.embeddings

        except Exception as e:
            raise ProviderError("cohere", f"Embedding failed: {e}") from e

    async def rerank(
        self,
        query: str,
        documents: List[str],
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by relevance to a query.

        Args:
            query: Search query
            documents: List of documents to rerank
            model: Rerank model (default: rerank-english-v3.0)
            top_n: Number of top results to return

        Returns:
            List of reranked results with scores
        """
        self._ensure_initialized()

        rerank_model = model or "rerank-english-v3.0"

        try:
            params = {
                "query": query,
                "documents": documents,
                "model": rerank_model,
            }

            if top_n:
                params["top_n"] = top_n

            response = await self._async_client.rerank(**params)

            return [
                {
                    "index": result.index,
                    "document": documents[result.index],
                    "relevance_score": result.relevance_score,
                }
                for result in response.results
            ]

        except Exception as e:
            raise ProviderError("cohere", f"Rerank failed: {e}") from e

    def _handle_error(self, error: Exception) -> None:
        """Handle Cohere API errors."""
        error_msg = str(error)

        if "401" in error_msg or "invalid api key" in error_msg.lower():
            raise AuthenticationError("cohere", error_msg) from error
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            raise RateLimitError("cohere", retry_after=60) from error
        elif "400" in error_msg:
            if "token" in error_msg.lower():
                raise TokenLimitExceededError("cohere", 0, 0) from error
            raise InvalidResponseError("cohere", error_msg) from error
        elif "404" in error_msg or "model" in error_msg.lower():
            raise InvalidModelError(self._model, "cohere") from error
        else:
            raise ProviderError("cohere", error_msg) from error


# Register the provider
register_provider("cohere", ProviderType.LLM, CohereProvider)
