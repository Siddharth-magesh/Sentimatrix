"""
Unit tests for LLM Provider Manager.

Tests cover:
- Manager initialization
- Adding/removing providers
- Generate with fallback
- Streaming
- Function calling
- Embeddings
- Health tracking
- Fallback strategies
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    LLMProviderError,
    ProviderNotFoundError,
    RateLimitError,
)
from sentimatrix.providers.base import LLMResponse, ProviderType, TokenUsage


# Mock response classes
@dataclass
class MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


class MockLLMResponse:
    """Mock LLM response."""

    def __init__(
        self,
        content: str = "Mock response",
        model: str = "mock-model",
        provider: str = "mock",
    ):
        self.content = content
        self.model = model
        self.provider = provider
        self.usage = TokenUsage(10, 20, 30)
        self.finish_reason = "stop"
        self.response_time_ms = 100.0
        self.tool_calls = None
        self.raw_response = None


class MockProvider:
    """Mock LLM provider for testing."""

    def __init__(self, name: str = "mock", should_fail: bool = False, fail_count: int = 0):
        self.name = name
        self.should_fail = should_fail
        self.fail_count = fail_count
        self._call_count = 0
        self._initialized = False

        # Mock capabilities
        self.info = MagicMock()
        self.info.capabilities.function_calling = True
        self.info.capabilities.embeddings = True

    @property
    def supports_function_calling(self):
        return True

    async def initialize(self):
        self._initialized = True

    async def close(self):
        self._initialized = False

    async def generate(self, prompt, **kwargs):
        self._call_count += 1
        if self.should_fail or self._call_count <= self.fail_count:
            raise LLMProviderError(f"Mock failure {self._call_count}", provider=self.name)
        return MockLLMResponse(content=f"Response from {self.name}", provider=self.name)

    async def generate_stream(self, prompt, **kwargs):
        if self.should_fail:
            raise LLMProviderError("Mock stream failure", provider=self.name)
        for chunk in ["Hello", " ", "World"]:
            yield chunk

    async def generate_with_functions(self, prompt, functions, **kwargs):
        if self.should_fail:
            raise LLMProviderError("Mock function failure", provider=self.name)
        return MockLLMResponse(provider=self.name)

    async def embed(self, text):
        if self.should_fail:
            raise LLMProviderError("Mock embed failure", provider=self.name)
        return [0.1, 0.2, 0.3]


class TestLLMManagerInit:
    """Test LLM Manager initialization."""

    def test_init_default(self):
        """Test default initialization."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        assert len(manager._providers) == 0
        assert not manager._initialized

    def test_add_provider(self):
        """Test adding a provider."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")

        manager.add_provider("openai", config, priority=0)

        assert "openai" in manager._provider_configs
        assert "openai" in manager._health

    def test_remove_provider(self):
        """Test removing a provider."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")

        manager.add_provider("openai", config)
        manager.remove_provider("openai")

        assert "openai" not in manager._provider_configs


class TestLLMManagerGenerate:
    """Test LLM Manager generate method."""

    @pytest.mark.asyncio
    async def test_generate_with_mock_provider(self):
        """Test generate with a mock provider."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        mock_provider = MockProvider("test")

        # Manually add mock provider
        manager._providers["test"] = mock_provider
        manager._provider_configs["test"] = MagicMock()
        manager._provider_configs["test"].max_retries = 2
        manager._health["test"] = MagicMock()
        manager._health["test"].is_available.return_value = True
        manager._health["test"].record_success = MagicMock()
        manager._initialized = True

        response = await manager.generate("Hello!")

        assert response.content == "Response from test"
        assert response.provider == "test"

    @pytest.mark.asyncio
    async def test_generate_specific_provider(self):
        """Test generate with a specific provider."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        mock_provider1 = MockProvider("provider1")
        mock_provider2 = MockProvider("provider2")

        manager._providers = {"provider1": mock_provider1, "provider2": mock_provider2}
        manager._provider_configs = {
            "provider1": MagicMock(max_retries=2),
            "provider2": MagicMock(max_retries=2),
        }
        manager._health = {
            "provider1": MagicMock(is_available=MagicMock(return_value=True), record_success=MagicMock()),
            "provider2": MagicMock(is_available=MagicMock(return_value=True), record_success=MagicMock()),
        }
        manager._initialized = True

        response = await manager.generate("Hello!", provider_name="provider2")

        assert response.provider == "provider2"

    @pytest.mark.asyncio
    async def test_generate_provider_not_found(self):
        """Test generate with non-existent provider."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        manager._initialized = True

        with pytest.raises(ProviderNotFoundError):
            await manager.generate("Hello!", provider_name="nonexistent")


class TestLLMManagerFallback:
    """Test LLM Manager fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test automatic fallback when first provider fails."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()

        failing_provider = MockProvider("failing", should_fail=True)
        working_provider = MockProvider("working")

        manager._providers = {"failing": failing_provider, "working": working_provider}
        manager._provider_configs = {
            "failing": MagicMock(max_retries=1, priority=0),
            "working": MagicMock(max_retries=1, priority=1),
        }
        manager._health = {
            "failing": MagicMock(is_available=MagicMock(return_value=True), record_failure=MagicMock(), record_success=MagicMock()),
            "working": MagicMock(is_available=MagicMock(return_value=True), record_failure=MagicMock(), record_success=MagicMock()),
        }
        manager._initialized = True

        response = await manager.generate("Hello!")

        assert response.provider == "working"

    @pytest.mark.asyncio
    async def test_all_providers_fail(self):
        """Test error when all providers fail."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()

        failing1 = MockProvider("failing1", should_fail=True)
        failing2 = MockProvider("failing2", should_fail=True)

        manager._providers = {"failing1": failing1, "failing2": failing2}
        manager._provider_configs = {
            "failing1": MagicMock(max_retries=1, priority=0),
            "failing2": MagicMock(max_retries=1, priority=1),
        }
        manager._health = {
            "failing1": MagicMock(is_available=MagicMock(return_value=True), record_failure=MagicMock()),
            "failing2": MagicMock(is_available=MagicMock(return_value=True), record_failure=MagicMock()),
        }
        manager._initialized = True

        with pytest.raises(LLMProviderError) as exc_info:
            await manager.generate("Hello!")

        assert "All providers failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retry_before_fallback(self):
        """Test retries before falling back."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()

        # Provider fails first 2 times, then succeeds
        flaky_provider = MockProvider("flaky", fail_count=2)

        manager._providers = {"flaky": flaky_provider}
        manager._provider_configs = {
            "flaky": MagicMock(max_retries=3, priority=0),
        }
        manager._health = {
            "flaky": MagicMock(is_available=MagicMock(return_value=True), record_failure=MagicMock(), record_success=MagicMock()),
        }
        manager._initialized = True

        response = await manager.generate("Hello!")

        assert response.provider == "flaky"
        assert flaky_provider._call_count == 3  # 2 failures + 1 success


class TestLLMManagerStream:
    """Test LLM Manager streaming."""

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming generation."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        mock_provider = MockProvider("test")

        manager._providers = {"test": mock_provider}
        manager._provider_configs = {"test": MagicMock(priority=0)}
        manager._health = {"test": MagicMock(is_available=MagicMock(return_value=True))}
        manager._initialized = True

        chunks = []
        async for chunk in manager.generate_stream("Hello!"):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello World"


class TestLLMManagerFunctions:
    """Test LLM Manager function calling."""

    @pytest.mark.asyncio
    async def test_generate_with_functions(self):
        """Test function calling through manager."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        mock_provider = MockProvider("test")

        manager._providers = {"test": mock_provider}
        manager._provider_configs = {"test": MagicMock(priority=0)}
        manager._health = {"test": MagicMock(is_available=MagicMock(return_value=True), record_success=MagicMock())}
        manager._initialized = True

        functions = [{"name": "test", "description": "test"}]
        response = await manager.generate_with_functions("Hello!", functions)

        assert response.provider == "test"


class TestLLMManagerEmbed:
    """Test LLM Manager embeddings."""

    @pytest.mark.asyncio
    async def test_embed(self):
        """Test embedding through manager."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        mock_provider = MockProvider("test")

        manager._providers = {"test": mock_provider}
        manager._provider_configs = {"test": MagicMock(priority=0)}
        manager._health = {"test": MagicMock(is_available=MagicMock(return_value=True))}
        manager._initialized = True

        embedding = await manager.embed("Hello!")

        assert embedding == [0.1, 0.2, 0.3]


class TestLLMManagerHealth:
    """Test LLM Manager health tracking."""

    def test_provider_health_record_success(self):
        """Test recording successful requests."""
        from sentimatrix.providers.llm.manager import ProviderHealth

        health = ProviderHealth(provider_name="test")

        health.record_success(100.0)

        assert health.is_healthy
        assert health.consecutive_failures == 0
        assert health.total_requests == 1
        assert health.avg_response_time_ms == 100.0

    def test_provider_health_record_failure(self):
        """Test recording failed requests."""
        from sentimatrix.providers.llm.manager import ProviderHealth

        health = ProviderHealth(provider_name="test")

        health.record_failure("Error")
        health.record_failure("Error")
        health.record_failure("Error")

        assert not health.is_healthy
        assert health.consecutive_failures == 3
        assert health.total_failures == 3

    def test_provider_health_rate_limit(self):
        """Test rate limit handling."""
        import time
        from sentimatrix.providers.llm.manager import ProviderHealth

        health = ProviderHealth(provider_name="test")

        health.record_failure("Rate limit", is_rate_limit=True)

        assert health.rate_limited_until is not None
        assert health.rate_limited_until > time.time()
        assert not health.is_available()

    def test_get_health_status(self):
        """Test getting health status of all providers."""
        from sentimatrix.providers.llm.manager import LLMProviderManager, ProviderHealth

        manager = LLMProviderManager()
        manager._health = {
            "provider1": ProviderHealth(provider_name="provider1"),
            "provider2": ProviderHealth(provider_name="provider2"),
        }

        status = manager.get_health_status()

        assert "provider1" in status
        assert "provider2" in status


class TestLLMManagerFallbackStrategies:
    """Test different fallback strategies."""

    def test_sequential_strategy(self):
        """Test sequential fallback strategy."""
        from sentimatrix.providers.llm.manager import LLMProviderManager, FallbackStrategy, ProviderHealth

        manager = LLMProviderManager()
        manager._config.fallback_strategy = FallbackStrategy.SEQUENTIAL

        manager._providers = {"a": MagicMock(), "b": MagicMock(), "c": MagicMock()}
        manager._provider_configs = {
            "a": MagicMock(priority=2),
            "b": MagicMock(priority=0),
            "c": MagicMock(priority=1),
        }
        manager._health = {
            "a": ProviderHealth(provider_name="a"),
            "b": ProviderHealth(provider_name="b"),
            "c": ProviderHealth(provider_name="c"),
        }

        ordered = manager._get_ordered_providers()

        # Should be sorted by priority (0, 1, 2)
        assert ordered[0] == "b"  # priority 0
        assert ordered[1] == "c"  # priority 1
        assert ordered[2] == "a"  # priority 2

    def test_round_robin_strategy(self):
        """Test round-robin fallback strategy."""
        from sentimatrix.providers.llm.manager import LLMProviderManager, FallbackStrategy, ProviderHealth

        manager = LLMProviderManager()
        manager._config.fallback_strategy = FallbackStrategy.ROUND_ROBIN

        manager._providers = {"a": MagicMock(), "b": MagicMock(), "c": MagicMock()}
        manager._provider_configs = {
            "a": MagicMock(priority=0),
            "b": MagicMock(priority=1),
            "c": MagicMock(priority=2),
        }
        manager._health = {
            "a": ProviderHealth(provider_name="a"),
            "b": ProviderHealth(provider_name="b"),
            "c": ProviderHealth(provider_name="c"),
        }

        # Get providers multiple times - should rotate
        first_call = manager._get_ordered_providers()
        second_call = manager._get_ordered_providers()

        # The order should change (rotate)
        assert first_call[0] != second_call[0] or len(first_call) == 1

    def test_fastest_strategy(self):
        """Test fastest-first fallback strategy."""
        from sentimatrix.providers.llm.manager import LLMProviderManager, FallbackStrategy, ProviderHealth

        manager = LLMProviderManager()
        manager._config.fallback_strategy = FallbackStrategy.FASTEST

        manager._providers = {"slow": MagicMock(), "fast": MagicMock(), "medium": MagicMock()}
        manager._provider_configs = {
            "slow": MagicMock(),
            "fast": MagicMock(),
            "medium": MagicMock(),
        }

        # Set different response times
        health_slow = ProviderHealth(provider_name="slow")
        health_slow.avg_response_time_ms = 500.0

        health_fast = ProviderHealth(provider_name="fast")
        health_fast.avg_response_time_ms = 50.0

        health_medium = ProviderHealth(provider_name="medium")
        health_medium.avg_response_time_ms = 200.0

        manager._health = {
            "slow": health_slow,
            "fast": health_fast,
            "medium": health_medium,
        }

        ordered = manager._get_ordered_providers()

        # Should be sorted by response time (fastest first)
        assert ordered[0] == "fast"


class TestLLMManagerContextManager:
    """Test LLM Manager context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()

        async with manager:
            assert manager._initialized

        assert not manager._initialized


class TestCreateManagerFromConfig:
    """Test creating manager from configuration dict."""

    def test_create_from_config(self):
        """Test creating manager from config dict."""
        from sentimatrix.providers.llm.manager import create_manager_from_config

        config = {
            "providers": [
                {"name": "openai", "provider": "openai", "api_key": "sk-test", "model": "gpt-4o-mini"},
                {"name": "groq", "provider": "groq", "api_key": "gsk_test", "model": "llama-3.3-70b"},
            ],
            "fallback_strategy": "sequential"
        }

        manager = create_manager_from_config(config)

        assert "openai" in manager._provider_configs
        assert "groq" in manager._provider_configs

    def test_list_providers(self):
        """Test listing providers."""
        from sentimatrix.providers.llm.manager import LLMProviderManager

        manager = LLMProviderManager()
        config1 = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
        config2 = LLMConfig(provider="groq", model="llama-3.3-70b", api_key="gsk_test")

        manager.add_provider("openai", config1)
        manager.add_provider("groq", config2)

        providers = manager.list_providers()

        assert "openai" in providers
        assert "groq" in providers
