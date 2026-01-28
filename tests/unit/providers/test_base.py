"""
Unit Tests for Base Provider Interfaces

Tests the provider base classes and registry including:
- Provider capabilities
- Provider info
- Data models (LLMResponse, ScrapedContent, etc.)
- Provider registry
- Base class contracts
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from sentimatrix.core.config import LLMConfig, ModelConfig, ScraperConfig
from sentimatrix.core.exceptions import ProviderInitializationError, ProviderNotFoundError
from sentimatrix.providers.base import (
    BaseLLMProvider,
    BaseModelProvider,
    BaseProvider,
    BaseScraperProvider,
    LLMResponse,
    PredictionResult,
    ProviderCapabilities,
    ProviderInfo,
    ProviderRegistry,
    ProviderType,
    Review,
    ScrapedContent,
    TokenUsage,
    get_provider,
    list_providers,
    register_provider,
)


class TestProviderCapabilities:
    """Tests for ProviderCapabilities."""

    def test_default_capabilities(self):
        """Test default capability values."""
        caps = ProviderCapabilities()
        assert caps.streaming is False
        assert caps.function_calling is False
        assert caps.vision is False
        assert caps.json_mode is False
        assert caps.max_context_tokens == 4096

    def test_custom_capabilities(self):
        """Test custom capability values."""
        caps = ProviderCapabilities(
            streaming=True,
            function_calling=True,
            vision=True,
            max_context_tokens=128000,
        )
        assert caps.streaming is True
        assert caps.function_calling is True
        assert caps.vision is True
        assert caps.max_context_tokens == 128000

    def test_to_dict(self):
        """Test capabilities serialization."""
        caps = ProviderCapabilities(streaming=True)
        caps_dict = caps.to_dict()

        assert isinstance(caps_dict, dict)
        assert caps_dict["streaming"] is True
        assert "max_context_tokens" in caps_dict


class TestProviderInfo:
    """Tests for ProviderInfo."""

    def test_basic_info(self):
        """Test basic provider info."""
        info = ProviderInfo(
            name="test_provider",
            provider_type=ProviderType.LLM,
        )
        assert info.name == "test_provider"
        assert info.provider_type == ProviderType.LLM
        assert info.version == "1.0.0"

    def test_full_info(self):
        """Test provider info with all fields."""
        caps = ProviderCapabilities(streaming=True)
        info = ProviderInfo(
            name="openai",
            provider_type=ProviderType.LLM,
            version="2.0.0",
            description="OpenAI API provider",
            capabilities=caps,
            supported_models=["gpt-4", "gpt-3.5-turbo"],
            website="https://openai.com",
        )

        assert info.description == "OpenAI API provider"
        assert "gpt-4" in info.supported_models
        assert info.website == "https://openai.com"

    def test_to_dict(self):
        """Test provider info serialization."""
        info = ProviderInfo(
            name="test",
            provider_type=ProviderType.LLM,
            supported_models=["model1"],
        )
        info_dict = info.to_dict()

        assert info_dict["name"] == "test"
        assert info_dict["provider_type"] == "llm"
        assert "capabilities" in info_dict


class TestTokenUsage:
    """Tests for TokenUsage."""

    def test_default_usage(self):
        """Test default token usage."""
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_usage(self):
        """Test custom token usage."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )
        assert usage.prompt_tokens == 100
        assert usage.total_tokens == 150

    def test_addition(self):
        """Test adding token usages together."""
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300)

        combined = usage1 + usage2
        assert combined.prompt_tokens == 300
        assert combined.completion_tokens == 150
        assert combined.total_tokens == 450


class TestLLMResponse:
    """Tests for LLMResponse."""

    def test_basic_response(self):
        """Test basic LLM response."""
        response = LLMResponse(
            content="Hello, world!",
            model="gpt-4",
            provider="openai",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
        assert response.content == "Hello, world!"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
        assert response.finish_reason == "stop"

    def test_response_with_tool_calls(self):
        """Test response with tool calls."""
        tool_call = {
            "id": "call_123",
            "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
        }
        response = LLMResponse(
            content="",
            model="gpt-4",
            provider="openai",
            usage=TokenUsage(),
            tool_calls=[tool_call],
        )
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1

    def test_to_dict(self):
        """Test response serialization."""
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            provider="openai",
            usage=TokenUsage(total_tokens=10),
        )
        response_dict = response.to_dict()

        assert response_dict["content"] == "Test"
        assert "usage" in response_dict
        assert response_dict["usage"]["total_tokens"] == 10


class TestScrapedContent:
    """Tests for ScrapedContent."""

    def test_basic_content(self):
        """Test basic scraped content."""
        content = ScrapedContent(
            url="https://example.com",
            content="Page content here",
        )
        assert content.url == "https://example.com"
        assert content.content == "Page content here"
        assert content.status_code == 200

    def test_full_content(self):
        """Test scraped content with all fields."""
        content = ScrapedContent(
            url="https://example.com",
            title="Example Page",
            content="Content",
            html="<html>...</html>",
            status_code=200,
            response_time_ms=150.5,
            headers={"Content-Type": "text/html"},
            provider="playwright",
        )
        assert content.title == "Example Page"
        assert content.html is not None
        assert content.response_time_ms == 150.5
        assert content.provider == "playwright"

    def test_to_dict(self):
        """Test content serialization."""
        content = ScrapedContent(url="https://example.com", content="Test")
        content_dict = content.to_dict()

        assert content_dict["url"] == "https://example.com"
        assert "scraped_at" in content_dict


class TestReview:
    """Tests for Review dataclass."""

    def test_basic_review(self):
        """Test basic review creation."""
        review = Review(
            id="rev_123",
            text="Great product!",
            source="https://amazon.com/product/123",
            platform="amazon",
        )
        assert review.id == "rev_123"
        assert review.text == "Great product!"
        assert review.platform == "amazon"

    def test_review_with_metadata(self):
        """Test review with all fields."""
        review = Review(
            id="rev_456",
            text="Good quality",
            source="https://example.com",
            platform="custom",
            author="John Doe",
            rating=4.5,
            timestamp=datetime.now(),
            metadata={"verified": True, "helpful_votes": 10},
        )
        assert review.author == "John Doe"
        assert review.rating == 4.5
        assert review.metadata["verified"] is True

    def test_to_dict(self):
        """Test review serialization."""
        review = Review(
            id="rev_789",
            text="Test review",
            source="https://test.com",
            platform="test",
        )
        review_dict = review.to_dict()

        assert review_dict["id"] == "rev_789"
        assert review_dict["text"] == "Test review"


class TestPredictionResult:
    """Tests for PredictionResult."""

    def test_basic_prediction(self):
        """Test basic prediction result."""
        result = PredictionResult(
            label="positive",
            score=0.95,
        )
        assert result.label == "positive"
        assert result.score == 0.95

    def test_prediction_with_all_scores(self):
        """Test prediction with all label scores."""
        result = PredictionResult(
            label="positive",
            score=0.8,
            confidence=0.85,
            all_scores={"positive": 0.8, "negative": 0.15, "neutral": 0.05},
            model_name="bert-sentiment",
            processing_time_ms=50.5,
        )
        assert result.all_scores["negative"] == 0.15
        assert result.model_name == "bert-sentiment"

    def test_to_dict(self):
        """Test prediction serialization."""
        result = PredictionResult(label="neutral", score=0.6)
        result_dict = result.to_dict()

        assert result_dict["label"] == "neutral"
        assert result_dict["score"] == 0.6


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def test_singleton(self):
        """Test registry is singleton."""
        registry1 = ProviderRegistry()
        registry2 = ProviderRegistry()
        assert registry1 is registry2

    def test_register_class(self):
        """Test registering a provider class."""
        registry = ProviderRegistry()

        # Create a mock provider class
        class MockLLMProvider(BaseLLMProvider):
            @property
            def info(self):
                return ProviderInfo(name="mock", provider_type=ProviderType.LLM)

            async def initialize(self):
                self._initialized = True

            async def close(self):
                pass

            async def generate(self, prompt, **kwargs):
                return LLMResponse(
                    content="mock response",
                    model="mock",
                    provider="mock",
                    usage=TokenUsage(),
                )

            async def generate_stream(self, prompt, **kwargs):
                yield "mock"

        registry.register("mock_llm", ProviderType.LLM, MockLLMProvider)
        assert registry.is_registered("mock_llm", ProviderType.LLM)

    def test_register_factory(self):
        """Test registering a factory function."""
        registry = ProviderRegistry()

        def mock_factory(config=None, **kwargs):
            return MagicMock()

        registry.register_factory("mock_factory", ProviderType.SCRAPER, mock_factory)
        assert registry.is_registered("mock_factory", ProviderType.SCRAPER)

    def test_get_provider_not_found(self):
        """Test getting unregistered provider raises error."""
        registry = ProviderRegistry()

        with pytest.raises(ProviderNotFoundError):
            registry.get("nonexistent", ProviderType.LLM)

    def test_list_providers(self):
        """Test listing registered providers."""
        registry = ProviderRegistry()
        providers = registry.list_providers()

        assert isinstance(providers, dict)
        assert "llm" in providers
        assert "scraper" in providers
        assert "model" in providers

    def test_list_providers_filtered(self):
        """Test listing providers with type filter."""
        registry = ProviderRegistry()
        providers = registry.list_providers(ProviderType.LLM)

        assert "llm" in providers
        assert "scraper" not in providers

    def test_is_registered(self):
        """Test checking if provider is registered."""
        registry = ProviderRegistry()
        assert not registry.is_registered("definitely_not_registered", ProviderType.LLM)


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_provider_not_found(self):
        """Test get_provider raises error for unknown provider."""
        with pytest.raises(ProviderNotFoundError):
            get_provider("unknown_provider", "llm")

    def test_list_providers_function(self):
        """Test list_providers module function."""
        providers = list_providers()
        assert isinstance(providers, dict)


class TestBaseLLMProvider:
    """Tests for BaseLLMProvider abstract class."""

    def test_supports_properties(self):
        """Test support checking properties work with capabilities."""

        class TestProvider(BaseLLMProvider):
            @property
            def info(self):
                return ProviderInfo(
                    name="test",
                    provider_type=ProviderType.LLM,
                    capabilities=ProviderCapabilities(
                        streaming=True,
                        vision=False,
                        function_calling=True,
                    ),
                )

            async def initialize(self):
                self._initialized = True

            async def close(self):
                pass

            async def generate(self, prompt, **kwargs):
                return LLMResponse(
                    content="test", model="test", provider="test", usage=TokenUsage()
                )

            async def generate_stream(self, prompt, **kwargs):
                yield "test"

        provider = TestProvider()
        assert provider.supports_streaming is True
        assert provider.supports_vision is False
        assert provider.supports_function_calling is True

    def test_count_tokens_default(self):
        """Test default token counting."""

        class TestProvider(BaseLLMProvider):
            @property
            def info(self):
                return ProviderInfo(name="test", provider_type=ProviderType.LLM)

            async def initialize(self):
                self._initialized = True

            async def close(self):
                pass

            async def generate(self, prompt, **kwargs):
                return LLMResponse(
                    content="test", model="test", provider="test", usage=TokenUsage()
                )

            async def generate_stream(self, prompt, **kwargs):
                yield "test"

        provider = TestProvider()
        # Default implementation: ~4 characters per token
        count = provider.count_tokens("Hello world")
        assert count > 0


class TestBaseScraperProvider:
    """Tests for BaseScraperProvider abstract class."""

    def test_supports_platform(self):
        """Test platform support checking."""

        class TestScraper(BaseScraperProvider):
            @property
            def info(self):
                return ProviderInfo(name="test", provider_type=ProviderType.SCRAPER)

            async def initialize(self):
                self._initialized = True

            async def close(self):
                pass

            async def scrape(self, url, **kwargs):
                return ScrapedContent(url=url, content="")

            async def scrape_reviews(self, url, **kwargs):
                return []

            def get_supported_platforms(self):
                return ["amazon", "steam"]

        scraper = TestScraper()
        assert scraper.supports_platform("amazon") is True
        assert scraper.supports_platform("AMAZON") is True  # Case insensitive
        assert scraper.supports_platform("ebay") is False


class TestBaseModelProvider:
    """Tests for BaseModelProvider abstract class."""

    def test_model_name_property(self):
        """Test model name property."""

        class TestModel(BaseModelProvider):
            @property
            def info(self):
                return ProviderInfo(name="test", provider_type=ProviderType.MODEL)

            async def initialize(self):
                self._initialized = True

            async def close(self):
                pass

            async def predict(self, text, **kwargs):
                return PredictionResult(label="positive", score=0.9)

            async def predict_batch(self, texts, **kwargs):
                return [PredictionResult(label="positive", score=0.9) for _ in texts]

            def get_model_info(self):
                return {"name": "test-model"}

        config = ModelConfig(sentiment_model="custom-sentiment-model")
        provider = TestModel(config)
        assert "custom-sentiment-model" in provider.model_name


class TestBaseProviderContextManager:
    """Tests for async context manager protocol."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test provider can be used as context manager."""

        class TestProvider(BaseLLMProvider):
            @property
            def info(self):
                return ProviderInfo(name="test", provider_type=ProviderType.LLM)

            async def initialize(self):
                self._initialized = True

            async def close(self):
                self._initialized = False

            async def generate(self, prompt, **kwargs):
                return LLMResponse(
                    content="test", model="test", provider="test", usage=TokenUsage()
                )

            async def generate_stream(self, prompt, **kwargs):
                yield "test"

        async with TestProvider() as provider:
            assert provider.is_initialized is True

        assert provider.is_initialized is False

    @pytest.mark.asyncio
    async def test_ensure_initialized(self):
        """Test _ensure_initialized raises error if not initialized."""

        class TestProvider(BaseLLMProvider):
            @property
            def info(self):
                return ProviderInfo(name="test", provider_type=ProviderType.LLM)

            async def initialize(self):
                self._initialized = True

            async def close(self):
                pass

            async def generate(self, prompt, **kwargs):
                self._ensure_initialized()
                return LLMResponse(
                    content="test", model="test", provider="test", usage=TokenUsage()
                )

            async def generate_stream(self, prompt, **kwargs):
                yield "test"

        provider = TestProvider()
        with pytest.raises(ProviderInitializationError):
            provider._ensure_initialized()
