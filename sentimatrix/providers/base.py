"""
Sentimatrix Base Provider Interfaces

Defines abstract base classes and protocols for all provider types:
- BaseLLMProvider: For LLM API providers
- BaseScraperProvider: For web scraping providers
- BaseModelProvider: For ML model providers

Also includes provider registry for dynamic provider discovery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Type,
    TypeVar,
    Union,
)

from sentimatrix.core.config import LLMConfig, ModelConfig, ScraperConfig
from sentimatrix.core.exceptions import (
    ProviderInitializationError,
    ProviderNotFoundError,
)


class ProviderType(str, Enum):
    """Types of providers."""

    LLM = "llm"
    SCRAPER = "scraper"
    MODEL = "model"


@dataclass
class ProviderCapabilities:
    """Describes what a provider can do."""

    # LLM capabilities
    streaming: bool = False
    function_calling: bool = False
    vision: bool = False
    json_mode: bool = False
    embeddings: bool = False

    # Context limits
    max_context_tokens: int = 4096
    max_output_tokens: int = 4096

    # Scraper capabilities
    javascript_rendering: bool = False
    screenshots: bool = False
    pdf_generation: bool = False
    proxy_support: bool = False

    # Model capabilities
    batch_processing: bool = False
    gpu_support: bool = False
    quantization: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "streaming": self.streaming,
            "function_calling": self.function_calling,
            "vision": self.vision,
            "json_mode": self.json_mode,
            "embeddings": self.embeddings,
            "max_context_tokens": self.max_context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "javascript_rendering": self.javascript_rendering,
            "screenshots": self.screenshots,
            "pdf_generation": self.pdf_generation,
            "proxy_support": self.proxy_support,
            "batch_processing": self.batch_processing,
            "gpu_support": self.gpu_support,
            "quantization": self.quantization,
        }


@dataclass
class ProviderInfo:
    """Metadata about a provider."""

    name: str
    provider_type: ProviderType
    version: str = "1.0.0"
    description: str = ""
    capabilities: ProviderCapabilities = field(default_factory=ProviderCapabilities)
    supported_models: List[str] = field(default_factory=list)
    website: Optional[str] = None
    documentation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider_type": self.provider_type.value,
            "version": self.version,
            "description": self.description,
            "capabilities": self.capabilities.to_dict(),
            "supported_models": self.supported_models,
            "website": self.website,
            "documentation": self.documentation,
        }


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        """Add two token usages together."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    provider: str
    usage: TokenUsage
    finish_reason: str = "stop"
    response_time_ms: float = 0.0
    raw_response: Optional[Dict[str, Any]] = None

    # Function calling
    tool_calls: Optional[List[Dict[str, Any]]] = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
            "finish_reason": self.finish_reason,
            "response_time_ms": self.response_time_ms,
            "tool_calls": self.tool_calls,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ScrapedContent:
    """Content scraped from a URL."""

    url: str
    title: Optional[str] = None
    content: str = ""
    html: Optional[str] = None
    status_code: int = 200
    response_time_ms: float = 0.0

    # Metadata
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    scraped_at: datetime = field(default_factory=datetime.now)

    # Provider info
    provider: str = ""
    proxy_used: Optional[str] = None
    user_agent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "html": self.html,
            "status_code": self.status_code,
            "response_time_ms": self.response_time_ms,
            "provider": self.provider,
            "scraped_at": self.scraped_at.isoformat(),
        }


@dataclass
class Review:
    """Represents a scraped review."""

    id: str
    text: str
    source: str
    platform: str
    author: Optional[str] = None
    rating: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "platform": self.platform,
            "author": self.author,
            "rating": self.rating,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }


@dataclass
class PredictionResult:
    """Result from a model prediction."""

    label: str
    score: float
    confidence: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    processing_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "score": self.score,
            "confidence": self.confidence,
            "all_scores": self.all_scores,
            "model_name": self.model_name,
            "processing_time_ms": self.processing_time_ms,
        }


class BaseProvider(ABC):
    """Base class for all providers."""

    def __init__(self, config: Any = None) -> None:
        """
        Initialize provider.

        Args:
            config: Provider-specific configuration
        """
        self._config = config
        self._initialized = False

    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Get provider information."""
        pass

    @property
    def name(self) -> str:
        """Get provider name."""
        return self.info.name

    @property
    def is_initialized(self) -> bool:
        """Check if provider is initialized."""
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (load resources, verify credentials, etc.)."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Cleanup provider resources."""
        pass

    async def __aenter__(self) -> "BaseProvider":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure provider is initialized before operations."""
        if not self._initialized:
            raise ProviderInitializationError(
                self.name,
                "Provider not initialized. Call initialize() first.",
            )


class BaseLLMProvider(BaseProvider):
    """
    Abstract base class for LLM providers.

    All LLM providers must implement these methods:
    - generate: Generate a completion from a prompt
    - generate_stream: Stream a completion from a prompt

    Optional methods:
    - embed: Generate embeddings for text
    - count_tokens: Count tokens in text
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        """
        Initialize LLM provider.

        Args:
            config: LLM configuration
        """
        super().__init__(config)
        self._config: LLMConfig = config or LLMConfig()

    @abstractmethod
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
        Generate a completion from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with generated content
        """
        pass

    @abstractmethod
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
        Stream a completion from a prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (overrides config)
            max_tokens: Maximum tokens to generate (overrides config)
            stop: Stop sequences
            **kwargs: Additional provider-specific parameters

        Yields:
            Text chunks as they're generated
        """
        pass

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
            prompt: User prompt
            functions: List of function definitions
            system_prompt: Optional system prompt
            function_call: How to handle function calls ("auto", "none", or specific function)
            **kwargs: Additional parameters

        Returns:
            LLMResponse, potentially with tool_calls

        Raises:
            NotImplementedError: If provider doesn't support function calling
        """
        if not self.info.capabilities.function_calling:
            raise NotImplementedError(
                f"{self.name} does not support function calling"
            )
        raise NotImplementedError("Subclass must implement generate_with_functions")

    async def embed(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text.

        Args:
            text: Single text or list of texts

        Returns:
            Embedding vector(s)

        Raises:
            NotImplementedError: If provider doesn't support embeddings
        """
        if not self.info.capabilities.embeddings:
            raise NotImplementedError(f"{self.name} does not support embeddings")
        raise NotImplementedError("Subclass must implement embed")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens in

        Returns:
            Number of tokens

        Note:
            Default implementation provides rough estimate.
            Override for accurate counting.
        """
        # Rough estimate: ~4 characters per token for English
        return len(text) // 4

    @property
    def model(self) -> str:
        """Get configured model name."""
        return self._config.model

    @property
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return self.info.capabilities.streaming

    @property
    def supports_vision(self) -> bool:
        """Check if provider supports vision/image inputs."""
        return self.info.capabilities.vision

    @property
    def supports_function_calling(self) -> bool:
        """Check if provider supports function calling."""
        return self.info.capabilities.function_calling


class BaseScraperProvider(BaseProvider):
    """
    Abstract base class for scraper providers.

    All scraper providers must implement:
    - scrape: Scrape content from a URL
    - scrape_reviews: Extract reviews from a URL
    """

    def __init__(self, config: Optional[ScraperConfig] = None) -> None:
        """
        Initialize scraper provider.

        Args:
            config: Scraper configuration
        """
        super().__init__(config)
        self._config: ScraperConfig = config or ScraperConfig()

    @abstractmethod
    async def scrape(
        self,
        url: str,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ScrapedContent:
        """
        Scrape content from a URL.

        Args:
            url: URL to scrape
            wait_for: CSS selector to wait for before scraping
            timeout: Request timeout (overrides config)
            **kwargs: Additional provider-specific parameters

        Returns:
            ScrapedContent with page content
        """
        pass

    @abstractmethod
    async def scrape_reviews(
        self,
        url: str,
        limit: int = 100,
        sort_by: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Extract reviews from a URL.

        Args:
            url: URL to scrape reviews from
            limit: Maximum number of reviews to extract
            sort_by: Sort order (implementation-specific)
            **kwargs: Additional parameters

        Returns:
            List of Review objects
        """
        pass

    async def screenshot(
        self,
        url: str,
        path: str,
        full_page: bool = True,
        **kwargs: Any,
    ) -> str:
        """
        Take a screenshot of a page.

        Args:
            url: URL to screenshot
            path: Output file path
            full_page: Capture full page or viewport only
            **kwargs: Additional parameters

        Returns:
            Path to saved screenshot

        Raises:
            NotImplementedError: If provider doesn't support screenshots
        """
        if not self.info.capabilities.screenshots:
            raise NotImplementedError(f"{self.name} does not support screenshots")
        raise NotImplementedError("Subclass must implement screenshot")

    def get_supported_platforms(self) -> List[str]:
        """
        Get list of supported platforms/domains.

        Returns:
            List of platform names this scraper supports
        """
        return []

    def supports_platform(self, platform: str) -> bool:
        """
        Check if scraper supports a specific platform.

        Args:
            platform: Platform name to check

        Returns:
            True if platform is supported
        """
        supported = self.get_supported_platforms()
        if not supported:  # Empty means all platforms
            return True
        return platform.lower() in [p.lower() for p in supported]


class BaseModelProvider(BaseProvider):
    """
    Abstract base class for ML model providers (sentiment, emotion, etc.).

    All model providers must implement:
    - predict: Make a prediction on input
    - predict_batch: Make predictions on multiple inputs
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        """
        Initialize model provider.

        Args:
            config: Model configuration
        """
        super().__init__(config)
        self._config: ModelConfig = config or ModelConfig()
        self._model: Any = None

    @abstractmethod
    async def predict(self, text: str, **kwargs: Any) -> PredictionResult:
        """
        Make a prediction on input text.

        Args:
            text: Input text
            **kwargs: Additional parameters

        Returns:
            PredictionResult with label and scores
        """
        pass

    @abstractmethod
    async def predict_batch(
        self, texts: List[str], **kwargs: Any
    ) -> List[PredictionResult]:
        """
        Make predictions on multiple texts.

        Args:
            texts: List of input texts
            **kwargs: Additional parameters

        Returns:
            List of PredictionResult objects
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.

        Returns:
            Dictionary with model metadata
        """
        pass

    @property
    def model_name(self) -> str:
        """Get loaded model name."""
        return getattr(self._config, "sentiment_model", "unknown")

    @property
    def device(self) -> str:
        """Get device model is running on."""
        return self._config.device


# Type variable for generic provider
T = TypeVar("T", bound=BaseProvider)


class ProviderRegistry:
    """
    Registry for provider discovery and instantiation.

    Supports registration of provider classes and factory functions.
    """

    _instance: Optional["ProviderRegistry"] = None
    _providers: Dict[str, Dict[str, Type[BaseProvider]]] = {}
    _factories: Dict[str, Dict[str, Callable[..., BaseProvider]]] = {}

    def __new__(cls) -> "ProviderRegistry":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._providers = {
                "llm": {},
                "scraper": {},
                "model": {},
            }
            cls._factories = {
                "llm": {},
                "scraper": {},
                "model": {},
            }
        return cls._instance

    def register(
        self,
        name: str,
        provider_type: Union[str, ProviderType],
        provider_class: Type[BaseProvider],
    ) -> None:
        """
        Register a provider class.

        Args:
            name: Provider name (e.g., "openai", "playwright")
            provider_type: Type of provider ("llm", "scraper", "model")
            provider_class: Provider class to register
        """
        if isinstance(provider_type, ProviderType):
            provider_type = provider_type.value

        if provider_type not in self._providers:
            self._providers[provider_type] = {}

        self._providers[provider_type][name.lower()] = provider_class

    def register_factory(
        self,
        name: str,
        provider_type: Union[str, ProviderType],
        factory: Callable[..., BaseProvider],
    ) -> None:
        """
        Register a factory function for creating providers.

        Args:
            name: Provider name
            provider_type: Type of provider
            factory: Factory function that creates provider instances
        """
        if isinstance(provider_type, ProviderType):
            provider_type = provider_type.value

        if provider_type not in self._factories:
            self._factories[provider_type] = {}

        self._factories[provider_type][name.lower()] = factory

    def get(
        self,
        name: str,
        provider_type: Union[str, ProviderType],
        config: Any = None,
        **kwargs: Any,
    ) -> BaseProvider:
        """
        Get a provider instance by name.

        Args:
            name: Provider name
            provider_type: Type of provider
            config: Provider configuration
            **kwargs: Additional arguments for provider constructor

        Returns:
            Provider instance

        Raises:
            ProviderNotFoundError: If provider is not registered
        """
        if isinstance(provider_type, ProviderType):
            provider_type = provider_type.value

        name_lower = name.lower()

        # Check factories first
        if provider_type in self._factories and name_lower in self._factories[provider_type]:
            factory = self._factories[provider_type][name_lower]
            return factory(config, **kwargs)

        # Then check registered classes
        if provider_type in self._providers and name_lower in self._providers[provider_type]:
            provider_class = self._providers[provider_type][name_lower]
            return provider_class(config, **kwargs)

        raise ProviderNotFoundError(f"{provider_type}:{name}")

    def list_providers(
        self, provider_type: Optional[Union[str, ProviderType]] = None
    ) -> Dict[str, List[str]]:
        """
        List registered providers.

        Args:
            provider_type: Optional type filter

        Returns:
            Dictionary of provider types to provider names
        """
        if provider_type is not None:
            if isinstance(provider_type, ProviderType):
                provider_type = provider_type.value
            return {
                provider_type: list(self._providers.get(provider_type, {}).keys())
                + list(self._factories.get(provider_type, {}).keys())
            }

        result = {}
        for pt in ["llm", "scraper", "model"]:
            result[pt] = list(self._providers.get(pt, {}).keys()) + list(
                self._factories.get(pt, {}).keys()
            )
        return result

    def is_registered(
        self, name: str, provider_type: Union[str, ProviderType]
    ) -> bool:
        """
        Check if a provider is registered.

        Args:
            name: Provider name
            provider_type: Type of provider

        Returns:
            True if provider is registered
        """
        if isinstance(provider_type, ProviderType):
            provider_type = provider_type.value

        name_lower = name.lower()
        return (
            name_lower in self._providers.get(provider_type, {})
            or name_lower in self._factories.get(provider_type, {})
        )


# Module-level convenience functions


def get_provider(
    name: str,
    provider_type: Union[str, ProviderType],
    config: Any = None,
    **kwargs: Any,
) -> BaseProvider:
    """
    Get a provider instance by name.

    Args:
        name: Provider name (e.g., "openai", "playwright")
        provider_type: Type of provider ("llm", "scraper", "model")
        config: Provider configuration
        **kwargs: Additional arguments

    Returns:
        Provider instance

    Example:
        >>> provider = get_provider("openai", "llm", config=llm_config)
        >>> await provider.initialize()
    """
    registry = ProviderRegistry()
    return registry.get(name, provider_type, config, **kwargs)


def register_provider(
    name: str,
    provider_type: Union[str, ProviderType],
    provider_class: Type[BaseProvider],
) -> None:
    """
    Register a provider class.

    Args:
        name: Provider name
        provider_type: Type of provider
        provider_class: Provider class to register

    Example:
        >>> register_provider("custom", "llm", CustomLLMProvider)
    """
    registry = ProviderRegistry()
    registry.register(name, provider_type, provider_class)


def list_providers(
    provider_type: Optional[Union[str, ProviderType]] = None
) -> Dict[str, List[str]]:
    """
    List registered providers.

    Args:
        provider_type: Optional type filter

    Returns:
        Dictionary of provider types to provider names
    """
    registry = ProviderRegistry()
    return registry.list_providers(provider_type)
