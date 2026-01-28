"""
Unit tests for OpenAI LLM Provider.

Tests cover:
- Provider initialization and configuration
- Generate method with various parameters
- Streaming responses
- Function calling
- Embeddings
- Error handling
- Token counting
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any, List

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import (
    AuthenticationError,
    ContentFilteredError,
    InvalidModelError,
    OpenAIError,
    RateLimitError,
    TokenLimitExceededError,
)
from sentimatrix.providers.base import LLMResponse, ProviderType, TokenUsage


# Mock openai module
@dataclass
class MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


@dataclass
class MockMessage:
    content: str = "Hello! How can I help?"
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
    id: str = "chatcmpl-123"
    model: str = "gpt-4o-mini"
    choices: List[MockChoice] = None
    usage: MockUsage = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = [MockChoice()]
        if self.usage is None:
            self.usage = MockUsage()

    def model_dump(self):
        return {"id": self.id, "model": self.model}


@dataclass
class MockDelta:
    content: str = ""


@dataclass
class MockStreamChoice:
    delta: MockDelta = None
    finish_reason: str = None

    def __post_init__(self):
        if self.delta is None:
            self.delta = MockDelta()


@dataclass
class MockStreamChunk:
    choices: List[MockStreamChoice] = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = [MockStreamChoice()]


@dataclass
class MockEmbeddingData:
    embedding: List[float] = None

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = [0.1, 0.2, 0.3]


@dataclass
class MockEmbeddingResponse:
    data: List[MockEmbeddingData] = None

    def __post_init__(self):
        if self.data is None:
            self.data = [MockEmbeddingData()]


class TestOpenAIProviderInit:
    """Test OpenAI provider initialization."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key="sk-test-key",
                timeout=60,
            )
            provider = OpenAIProvider(config)

            assert provider._model == "gpt-4o-mini"
            assert provider._api_key == "sk-test-key"
            assert not provider._initialized

    def test_init_without_config(self):
        """Test initialization without configuration."""
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            provider = OpenAIProvider()
            assert provider._model == "gpt-4o-mini"
            assert provider._api_key is None

    def test_provider_info(self):
        """Test provider information."""
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini")
            provider = OpenAIProvider(config)

            info = provider.info
            assert info.name == "openai"
            assert info.provider_type == ProviderType.LLM
            assert info.capabilities.streaming is True
            assert info.capabilities.function_calling is True
            assert info.capabilities.vision is True

    def test_provider_info_gpt4o(self):
        """Test provider info for GPT-4o model."""
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o")
            provider = OpenAIProvider(config)

            info = provider.info
            assert info.capabilities.max_context_tokens == 128000
            assert info.capabilities.vision is True


class TestOpenAIProviderInitialize:
    """Test OpenAI provider initialization."""

    @pytest.mark.asyncio
    async def test_initialize_with_api_key(self):
        """Test initialization with API key in config."""
        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock()
        mock_openai.OpenAI = MagicMock()

        with patch.dict('sys.modules', {'openai': mock_openai}):
            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)

            await provider.initialize()

            assert provider._initialized
            mock_openai.AsyncOpenAI.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_with_env_key(self):
        """Test initialization with API key from environment."""
        import os

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock()
        mock_openai.OpenAI = MagicMock()

        with patch.dict('sys.modules', {'openai': mock_openai}):
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-env-key'}):
                from sentimatrix.providers.llm.openai_provider import OpenAIProvider

                config = LLMConfig(provider="openai", model="gpt-4o-mini")
                provider = OpenAIProvider(config)

                await provider.initialize()

                assert provider._initialized

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        import os

        mock_openai = MagicMock()

        with patch.dict('sys.modules', {'openai': mock_openai}):
            with patch.dict(os.environ, {}, clear=True):
                # Remove OPENAI_API_KEY if present
                os.environ.pop('OPENAI_API_KEY', None)

                from sentimatrix.providers.llm.openai_provider import OpenAIProvider

                config = LLMConfig(provider="openai", model="gpt-4o-mini")
                provider = OpenAIProvider(config)

                with pytest.raises(AuthenticationError):
                    await provider.initialize()


class TestOpenAIProviderGenerate:
    """Test OpenAI provider generate method."""

    @pytest.mark.asyncio
    async def test_generate_basic(self):
        """Test basic generate call."""
        mock_response = MockResponse()

        # Create proper async mock
        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(return_value=mock_response)

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        # Set up error types
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            # Reset the module's cached openai
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            response = await provider.generate("Hello!")

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello! How can I help?"
            assert response.model == "gpt-4o-mini"
            assert response.provider == "openai"
            assert response.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self):
        """Test generate with system prompt."""
        mock_response = MockResponse()

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(return_value=mock_response)

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            await provider.generate(
                "Hello!",
                system_prompt="You are a helpful assistant."
            )

            call_args = mock_completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 2
            assert messages[0]["role"] == "system"
            assert messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_generate_with_parameters(self):
        """Test generate with custom parameters."""
        mock_response = MockResponse()

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(return_value=mock_response)

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            await provider.generate(
                "Hello!",
                temperature=0.5,
                max_tokens=100,
                stop=["END"]
            )

            call_args = mock_completions.create.call_args
            assert call_args.kwargs["temperature"] == 0.5
            assert call_args.kwargs["max_tokens"] == 100
            assert call_args.kwargs["stop"] == ["END"]

    @pytest.mark.asyncio
    async def test_generate_not_initialized(self):
        """Test generate fails when not initialized."""
        mock_openai = MagicMock()

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider
            from sentimatrix.core.exceptions import ProviderInitializationError

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)

            with pytest.raises(ProviderInitializationError):
                await provider.generate("Hello!")


class TestOpenAIProviderStream:
    """Test OpenAI provider streaming."""

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming generation."""
        chunks = [
            MockStreamChunk(choices=[MockStreamChoice(delta=MockDelta(content="Hello"))]),
            MockStreamChunk(choices=[MockStreamChoice(delta=MockDelta(content=" "))]),
            MockStreamChunk(choices=[MockStreamChoice(delta=MockDelta(content="World"))]),
            MockStreamChunk(choices=[MockStreamChoice(delta=MockDelta(content=""), finish_reason="stop")]),
        ]

        async def async_chunks():
            for chunk in chunks:
                yield chunk

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(return_value=async_chunks())

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            result = []
            async for chunk in provider.generate_stream("Hello!"):
                result.append(chunk)

            assert "".join(result) == "Hello World"


class TestOpenAIProviderFunctions:
    """Test OpenAI provider function calling."""

    @pytest.mark.asyncio
    async def test_generate_with_functions(self):
        """Test function calling."""
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Paris"}'

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tool_call]

        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.model = "gpt-4o-mini"
        mock_response.choices = [mock_choice]
        mock_response.usage = MockUsage()
        mock_response.model_dump = MagicMock(return_value={})

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(return_value=mock_response)

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            functions = [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]

            response = await provider.generate_with_functions(
                "What's the weather?",
                functions
            )

            assert response.tool_calls is not None
            assert len(response.tool_calls) == 1
            assert response.tool_calls[0]["function"]["name"] == "get_weather"


class TestOpenAIProviderEmbed:
    """Test OpenAI provider embeddings."""

    @pytest.mark.asyncio
    async def test_embed_single(self):
        """Test single text embedding."""
        mock_response = MockEmbeddingResponse()

        mock_embeddings = AsyncMock()
        mock_embeddings.create = AsyncMock(return_value=mock_response)

        mock_client = AsyncMock()
        mock_client.embeddings = mock_embeddings
        mock_client.chat = MagicMock()
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            embedding = await provider.embed("Hello")

            assert isinstance(embedding, list)
            assert len(embedding) == 3

    @pytest.mark.asyncio
    async def test_embed_batch(self):
        """Test batch embedding."""
        mock_response = MockEmbeddingResponse(data=[
            MockEmbeddingData(embedding=[0.1, 0.2]),
            MockEmbeddingData(embedding=[0.3, 0.4]),
        ])

        mock_embeddings = AsyncMock()
        mock_embeddings.create = AsyncMock(return_value=mock_response)

        mock_client = AsyncMock()
        mock_client.embeddings = mock_embeddings
        mock_client.chat = MagicMock()
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            embeddings = await provider.embed(["Hello", "World"])

            assert isinstance(embeddings, list)
            assert len(embeddings) == 2


class TestOpenAIProviderErrors:
    """Test OpenAI provider error handling."""

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        """Test authentication error handling."""
        class MockAuthError(Exception):
            pass

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(side_effect=MockAuthError("Invalid API key"))

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = MockAuthError
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="invalid")
            provider = OpenAIProvider(config)
            await provider.initialize()

            with pytest.raises(AuthenticationError):
                await provider.generate("Hello!")

    @pytest.mark.asyncio
    async def test_rate_limit_error(self):
        """Test rate limit error handling."""
        class MockRateLimitError(Exception):
            pass

        mock_completions = AsyncMock()
        mock_completions.create = AsyncMock(side_effect=MockRateLimitError("Rate limit"))

        mock_chat = MagicMock()
        mock_chat.completions = mock_completions

        mock_client = AsyncMock()
        mock_client.chat = mock_chat
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = MockRateLimitError
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            with pytest.raises(RateLimitError):
                await provider.generate("Hello!")


class TestOpenAIProviderTokens:
    """Test OpenAI provider token counting."""

    def test_count_tokens_with_tiktoken(self):
        """Test token counting with tiktoken."""
        mock_encoding = MagicMock()
        mock_encoding.encode = MagicMock(return_value=[1, 2, 3, 4, 5])

        mock_tiktoken = MagicMock()
        mock_tiktoken.encoding_for_model = MagicMock(return_value=mock_encoding)

        mock_openai = MagicMock()

        with patch.dict('sys.modules', {'openai': mock_openai, 'tiktoken': mock_tiktoken}):
            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)

            count = provider.count_tokens("Hello world")
            assert count == 5

    def test_count_tokens_fallback(self):
        """Test token counting fallback without tiktoken."""
        mock_openai = MagicMock()

        with patch.dict('sys.modules', {'openai': mock_openai}):
            # Remove tiktoken from cache
            import sys
            sys.modules.pop('tiktoken', None)

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider, _get_tiktoken

            # Reset the global
            import sentimatrix.providers.llm.openai_provider as module
            module._tiktoken = None

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)

            # This should use the fallback estimate
            count = provider.count_tokens("Hello world test")
            # Fallback is len(text) // 4
            assert count == len("Hello world test") // 4


class TestOpenAIProviderClose:
    """Test OpenAI provider close method."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing provider."""
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")
            provider = OpenAIProvider(config)
            await provider.initialize()

            assert provider._initialized

            await provider.close()

            assert not provider._initialized
            mock_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        mock_sync_client = MagicMock()
        mock_sync_client.close = MagicMock()

        mock_openai = MagicMock()
        mock_openai.AsyncOpenAI = MagicMock(return_value=mock_client)
        mock_openai.OpenAI = MagicMock(return_value=mock_sync_client)
        mock_openai.AuthenticationError = type('AuthenticationError', (Exception,), {})
        mock_openai.RateLimitError = type('RateLimitError', (Exception,), {})
        mock_openai.BadRequestError = type('BadRequestError', (Exception,), {})
        mock_openai.NotFoundError = type('NotFoundError', (Exception,), {})
        mock_openai.APITimeoutError = type('APITimeoutError', (Exception,), {})
        mock_openai.APIConnectionError = type('APIConnectionError', (Exception,), {})
        mock_openai.OpenAIError = type('OpenAIError', (Exception,), {})

        with patch.dict('sys.modules', {'openai': mock_openai}):
            import sentimatrix.providers.llm.openai_provider as module
            module._openai = None

            from sentimatrix.providers.llm.openai_provider import OpenAIProvider

            config = LLMConfig(provider="openai", model="gpt-4o-mini", api_key="sk-test")

            async with OpenAIProvider(config) as provider:
                assert provider._initialized

            assert not provider._initialized
