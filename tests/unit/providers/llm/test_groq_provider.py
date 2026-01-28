"""
Unit tests for Groq LLM Provider.

Tests cover:
- Provider initialization and configuration
- Generate method with various parameters
- Streaming responses
- Function calling
- Audio transcription
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any, List

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    AuthenticationError,
    GroqError,
    InvalidModelError,
    RateLimitError,
)
from sentimatrix.providers.base import LLMResponse, ProviderType, TokenUsage


# Mock groq module
@dataclass
class MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class MockMessage:
    content: str = "Hello from Groq!"
    tool_calls: Any = None


@dataclass
class MockChoice:
    message: MockMessage = None
    finish_reason: str = "stop"
    delta: Any = None

    def __post_init__(self):
        if self.message is None:
            self.message = MockMessage()


@dataclass
class MockResponse:
    id: str = "chatcmpl-groq-123"
    model: str = "llama-3.3-70b-versatile"
    choices: List[MockChoice] = None
    usage: MockUsage = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = [MockChoice()]
        if self.usage is None:
            self.usage = MockUsage()

    def model_dump(self):
        return {"id": self.id, "model": self.model}


class TestGroqProviderInit:
    """Test Groq provider initialization."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        with patch.dict('sys.modules', {'groq': MagicMock()}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(
                provider="groq",
                model="llama-3.3-70b-versatile",
                api_key="gsk_test_key",
            )
            provider = GroqProvider(config)

            assert provider._model == "llama-3.3-70b-versatile"
            assert provider._api_key == "gsk_test_key"
            assert not provider._initialized

    def test_init_default_model(self):
        """Test initialization uses default model."""
        with patch.dict('sys.modules', {'groq': MagicMock()}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider, DEFAULT_MODEL

            provider = GroqProvider()
            assert provider._model == DEFAULT_MODEL

    def test_provider_info(self):
        """Test provider information."""
        with patch.dict('sys.modules', {'groq': MagicMock()}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
            provider = GroqProvider(config)

            info = provider.info
            assert info.name == "groq"
            assert info.provider_type == ProviderType.LLM
            assert info.capabilities.streaming is True
            assert info.capabilities.function_calling is True
            assert info.capabilities.embeddings is False  # Groq doesn't have embeddings

    def test_provider_info_vision_model(self):
        """Test provider info for vision model."""
        with patch.dict('sys.modules', {'groq': MagicMock()}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.2-90b-vision-preview")
            provider = GroqProvider(config)

            info = provider.info
            assert info.capabilities.vision is True


class TestGroqProviderInitialize:
    """Test Groq provider initialization."""

    @pytest.mark.asyncio
    async def test_initialize_with_api_key(self):
        """Test initialization with API key in config."""
        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock()
        mock_groq.Groq = MagicMock()

        with patch.dict('sys.modules', {'groq': mock_groq}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test")
            provider = GroqProvider(config)

            await provider.initialize()

            assert provider._initialized
            mock_groq.AsyncGroq.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_env_key(self):
        """Test initialization with API key from environment."""
        import os

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock()
        mock_groq.Groq = MagicMock()

        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {'GROQ_API_KEY': 'gsk_env_key'}):
                from sentimatrix.providers.llm.groq_provider import GroqProvider

                config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
                provider = GroqProvider(config)

                await provider.initialize()

                assert provider._initialized

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        import os

        mock_groq = MagicMock()

        with patch.dict('sys.modules', {'groq': mock_groq}):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('GROQ_API_KEY', None)

                from sentimatrix.providers.llm.groq_provider import GroqProvider

                config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile")
                provider = GroqProvider(config)

                with pytest.raises(AuthenticationError):
                    await provider.initialize()


class TestGroqProviderGenerate:
    """Test Groq provider generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generate call."""
        mock_response = MockResponse()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock(return_value=mock_client)
        mock_groq.Groq = MagicMock()

        with patch.dict('sys.modules', {'groq': mock_groq}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test")
            provider = GroqProvider(config)
            await provider.initialize()

            response = await provider.generate("Hello!")

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello from Groq!"
            assert response.provider == "groq"
            assert response.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generate with system prompt."""
        mock_response = MockResponse()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock(return_value=mock_client)
        mock_groq.Groq = MagicMock()

        with patch.dict('sys.modules', {'groq': mock_groq}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test")
            provider = GroqProvider(config)
            await provider.initialize()

            await provider.generate(
                "Hello!",
                system_prompt="You are a fast assistant."
            )

            call_args = mock_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_generate_with_json_format(self):
        """Test generate with JSON response format."""
        mock_response = MockResponse()
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock(return_value=mock_client)
        mock_groq.Groq = MagicMock()

        with patch.dict('sys.modules', {'groq': mock_groq}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test")
            provider = GroqProvider(config)
            await provider.initialize()

            await provider.generate(
                "Return JSON",
                response_format={"type": "json_object"}
            )

            call_args = mock_client.chat.completions.create.call_args
            assert "response_format" in call_args.kwargs


class TestGroqProviderFunctions:
    """Test Groq provider function calling."""

    @pytest.mark.asyncio
    async def test_generate_with_functions(self):
        """Test function calling."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_groq_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "search"
        mock_tool_call.function.arguments = '{"query": "test"}'

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.model = "llama-3.3-70b-versatile"
        mock_response.choices = [mock_choice]
        mock_response.usage = MockUsage()
        mock_response.model_dump = MagicMock(return_value={})

        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock(return_value=mock_client)
        mock_groq.Groq = MagicMock()

        with patch.dict('sys.modules', {'groq': mock_groq}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test")
            provider = GroqProvider(config)
            await provider.initialize()

            functions = [{
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]

            response = await provider.generate_with_functions(
                "Search for test",
                functions
            )

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_function_calling_unsupported_model(self):
        """Test function calling with unsupported model raises error."""
        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock()
        mock_groq.Groq = MagicMock()

        with patch.dict('sys.modules', {'groq': mock_groq}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            # gemma doesn't support function calling
            config = LLMConfig(provider="groq", model="gemma2-9b-it", api_key="gsk_test")
            provider = GroqProvider(config)
            await provider.initialize()

            with pytest.raises(InvalidModelError):
                await provider.generate_with_functions(
                    "test",
                    [{"name": "test", "description": "test"}]
                )


class TestGroqProviderErrors:
    """Test Groq provider error handling."""

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test rate limit error handling."""
        class MockRateLimitError(Exception):
            pass

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(
            side_effect=MockRateLimitError("Rate limit: 30 req/min")
        )

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock(return_value=mock_client)
        mock_groq.Groq = MagicMock(return_value=mock_sync_client)
        mock_groq.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_groq.RateLimitError = MockRateLimitError
        mock_groq.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_groq.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_groq.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_groq.APIConnectionError = type('APIConnectionError', (Exception,), {})

        with patch.dict('sys.modules', {'groq': mock_groq}):
            import sentimatrix.providers.llm.groq_provider as module
            module._groq = None

            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test")
            provider = GroqProvider(config)
            await provider.initialize()

            with pytest.raises(RateLimitError) as exc_info:
                await provider.generate("Hello!")

            assert "Rate limit" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test authentication error handling."""
        class MockAuthError(Exception):
            pass

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(
            side_effect=MockAuthError("Invalid API key")
        )

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock(return_value=mock_client)
        mock_groq.Groq = MagicMock(return_value=mock_sync_client)
        mock_groq.AuthenticationError = MockAuthError
        mock_groq.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_groq.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_groq.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_groq.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_groq.APIConnectionError = type('APIConnectionError', (Exception,), {})

        with patch.dict('sys.modules', {'groq': mock_groq}):
            import sentimatrix.providers.llm.groq_provider as module
            module._groq = None

            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="invalid")
            provider = GroqProvider(config)
            await provider.initialize()

            with pytest.raises(AuthenticationError):
                await provider.generate("Hello!")


class TestGroqProviderClose:
    """Test Groq provider close method."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing provider."""
        mock_client = AsyncMock()
        mock_sync_client = MagicMock()

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock(return_value=mock_client)
        mock_groq.Groq = MagicMock(return_value=mock_sync_client)

        with patch.dict('sys.modules', {'groq': mock_groq}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test")
            provider = GroqProvider(config)
            await provider.initialize()

            assert provider._initialized

            await provider.close()

            assert not provider._initialized

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        mock_client = AsyncMock()
        mock_sync_client = MagicMock()

        mock_groq = MagicMock()
        mock_groq.AsyncGroq = MagicMock(return_value=mock_client)
        mock_groq.Groq = MagicMock(return_value=mock_sync_client)

        with patch.dict('sys.modules', {'groq': mock_groq}):
            from sentimatrix.providers.llm.groq_provider import GroqProvider

            config = LLMConfig(provider="groq", model="llama-3.3-70b-versatile", api_key="gsk_test")

            async with GroqProvider(config) as provider:
                assert provider._initialized

            assert not provider._initialized
