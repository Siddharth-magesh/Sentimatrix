"""
Unit tests for new LLM Providers.

Tests cover:
- Azure OpenAI
- Amazon Bedrock
- Mistral
- Cerebras
- Fireworks AI
- Together AI
- OpenRouter
- Cohere
- LM Studio
- vLLM
- DeepSeek
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Any
import os

from sentimatrix.core.config import LLMConfig
from sentimatrix.core.exceptions import AuthenticationError
from sentimatrix.providers.base import LLMResponse, ProviderType, TokenUsage


# ============================================================================
# Mock Response Classes
# ============================================================================

@dataclass
class MockUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 20
    total_tokens: int = 30


class MockHTTPResponse:
    """Mock httpx response."""
    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.text = str(data)

    def json(self):
        return self._data


def create_mock_chat_response(content: str = "Hello!", model: str = "test-model"):
    """Create a standard chat completion response."""
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [{
            "message": {"content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        }
    }


# ============================================================================
# Azure OpenAI Provider Tests
# ============================================================================

class TestAzureOpenAIProvider:
    """Test Azure OpenAI provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            from sentimatrix.providers.llm.azure_openai_provider import AzureOpenAIProvider

            config = LLMConfig(
                provider="azure_openai",
                model="gpt-4o",
                api_key="test_key",
            )
            provider = AzureOpenAIProvider(config)

            assert provider._deployment == "gpt-4o"
            assert provider._api_key == "test_key"
            assert not provider._initialized

    def test_provider_info(self):
        """Test provider information."""
        with patch.dict('sys.modules', {'openai': MagicMock()}):
            from sentimatrix.providers.llm.azure_openai_provider import AzureOpenAIProvider

            config = LLMConfig(provider="azure_openai", model="gpt-4o")
            provider = AzureOpenAIProvider(config)

            info = provider.info
            assert info.name == "azure-openai"  # Uses hyphen in provider name
            assert info.provider_type == ProviderType.LLM
            assert info.capabilities.streaming is True

    @pytest.mark.asyncio
    async def test_initialize_without_endpoint(self):
        """Test initialization fails without endpoint."""
        mock_openai = MagicMock()

        with patch.dict('sys.modules', {'openai': mock_openai}):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('AZURE_OPENAI_ENDPOINT', None)
                os.environ.pop('AZURE_OPENAI_API_KEY', None)

                from sentimatrix.providers.llm.azure_openai_provider import AzureOpenAIProvider

                config = LLMConfig(provider="azure_openai", model="gpt-4o")
                provider = AzureOpenAIProvider(config)

                with pytest.raises(AuthenticationError):
                    await provider.initialize()


# ============================================================================
# Bedrock Provider Tests
# ============================================================================

class TestBedrockProvider:
    """Test Amazon Bedrock provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        with patch.dict('sys.modules', {'boto3': MagicMock()}):
            from sentimatrix.providers.llm.bedrock_provider import BedrockProvider

            config = LLMConfig(
                provider="bedrock",
                model="anthropic.claude-3-sonnet",
            )
            provider = BedrockProvider(config)

            assert provider._model == "anthropic.claude-3-sonnet"
            assert not provider._initialized

    def test_provider_info(self):
        """Test provider information."""
        with patch.dict('sys.modules', {'boto3': MagicMock()}):
            from sentimatrix.providers.llm.bedrock_provider import BedrockProvider

            config = LLMConfig(provider="bedrock", model="anthropic.claude-3-sonnet")
            provider = BedrockProvider(config)

            info = provider.info
            assert info.name == "bedrock"
            assert info.provider_type == ProviderType.LLM


# ============================================================================
# Mistral Provider Tests
# ============================================================================

class TestMistralProvider:
    """Test Mistral provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        with patch.dict('sys.modules', {'mistralai': MagicMock(), 'httpx': MagicMock()}):
            from sentimatrix.providers.llm.mistral_provider import MistralProvider

            config = LLMConfig(
                provider="mistral",
                model="mistral-large-latest",
                api_key="test_key",
            )
            provider = MistralProvider(config)

            assert provider._model == "mistral-large-latest"
            assert provider._api_key == "test_key"

    def test_provider_info(self):
        """Test provider information."""
        with patch.dict('sys.modules', {'mistralai': MagicMock(), 'httpx': MagicMock()}):
            from sentimatrix.providers.llm.mistral_provider import MistralProvider

            config = LLMConfig(provider="mistral", model="mistral-large-latest")
            provider = MistralProvider(config)

            info = provider.info
            assert info.name == "mistral"
            assert info.provider_type == ProviderType.LLM
            assert info.capabilities.streaming is True
            assert info.capabilities.embeddings is True

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        mock_mistral = MagicMock()
        mock_httpx = MagicMock()

        with patch.dict('sys.modules', {'mistralai': mock_mistral, 'httpx': mock_httpx}):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('MISTRAL_API_KEY', None)

                from sentimatrix.providers.llm.mistral_provider import MistralProvider

                config = LLMConfig(provider="mistral", model="mistral-large-latest")
                provider = MistralProvider(config)

                with pytest.raises(AuthenticationError):
                    await provider.initialize()


# ============================================================================
# Cerebras Provider Tests
# ============================================================================

class TestCerebrasProvider:
    """Test Cerebras provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.cerebras_provider import CerebrasProvider

        config = LLMConfig(
            provider="cerebras",
            model="llama3.1-70b",
            api_key="test_key",
        )
        provider = CerebrasProvider(config)

        assert provider._model == "llama3.1-70b"
        assert provider._api_key == "test_key"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.cerebras_provider import CerebrasProvider

        config = LLMConfig(provider="cerebras", model="llama3.1-70b")
        provider = CerebrasProvider(config)

        info = provider.info
        assert info.name == "cerebras"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert "1800" in info.description or "450" in info.description  # Speed info

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('CEREBRAS_API_KEY', None)

            from sentimatrix.providers.llm.cerebras_provider import CerebrasProvider

            config = LLMConfig(provider="cerebras", model="llama3.1-70b")
            provider = CerebrasProvider(config)

            with pytest.raises(AuthenticationError):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generate method."""
        mock_response = MockHTTPResponse(create_mock_chat_response("Hello from Cerebras!", "llama3.1-70b"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.cerebras_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.cerebras_provider import CerebrasProvider

            config = LLMConfig(provider="cerebras", model="llama3.1-70b", api_key="test_key")
            provider = CerebrasProvider(config)
            await provider.initialize()

            response = await provider.generate("Hello!")

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello from Cerebras!"
            assert response.provider == "cerebras"


# ============================================================================
# Fireworks Provider Tests
# ============================================================================

class TestFireworksProvider:
    """Test Fireworks AI provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.fireworks_provider import FireworksProvider

        config = LLMConfig(
            provider="fireworks",
            model="accounts/fireworks/models/llama-v3p1-70b-instruct",
            api_key="test_key",
        )
        provider = FireworksProvider(config)

        assert "llama-v3p1-70b" in provider._model
        assert provider._api_key == "test_key"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.fireworks_provider import FireworksProvider

        config = LLMConfig(provider="fireworks", model="accounts/fireworks/models/llama-v3p1-70b-instruct")
        provider = FireworksProvider(config)

        info = provider.info
        assert info.name == "fireworks"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert info.capabilities.embeddings is True

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('FIREWORKS_API_KEY', None)

            from sentimatrix.providers.llm.fireworks_provider import FireworksProvider

            config = LLMConfig(provider="fireworks", model="accounts/fireworks/models/llama-v3p1-70b-instruct")
            provider = FireworksProvider(config)

            with pytest.raises(AuthenticationError):
                await provider.initialize()


# ============================================================================
# Together Provider Tests
# ============================================================================

class TestTogetherProvider:
    """Test Together AI provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.together_provider import TogetherProvider

        config = LLMConfig(
            provider="together",
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            api_key="test_key",
        )
        provider = TogetherProvider(config)

        assert "Llama-3.3" in provider._model
        assert provider._api_key == "test_key"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.together_provider import TogetherProvider

        config = LLMConfig(provider="together", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
        provider = TogetherProvider(config)

        info = provider.info
        assert info.name == "together"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert info.capabilities.embeddings is True

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('TOGETHER_API_KEY', None)

            from sentimatrix.providers.llm.together_provider import TogetherProvider

            config = LLMConfig(provider="together", model="meta-llama/Llama-3.3-70B-Instruct-Turbo")
            provider = TogetherProvider(config)

            with pytest.raises(AuthenticationError):
                await provider.initialize()


# ============================================================================
# OpenRouter Provider Tests
# ============================================================================

class TestOpenRouterProvider:
    """Test OpenRouter provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.openrouter_provider import OpenRouterProvider

        config = LLMConfig(
            provider="openrouter",
            model="anthropic/claude-3.5-sonnet",
            api_key="test_key",
        )
        provider = OpenRouterProvider(config)

        assert "claude-3.5" in provider._model
        assert provider._api_key == "test_key"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.openrouter_provider import OpenRouterProvider

        config = LLMConfig(provider="openrouter", model="anthropic/claude-3.5-sonnet")
        provider = OpenRouterProvider(config)

        info = provider.info
        assert info.name == "openrouter"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert "200+" in info.description

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('OPENROUTER_API_KEY', None)

            from sentimatrix.providers.llm.openrouter_provider import OpenRouterProvider

            config = LLMConfig(provider="openrouter", model="anthropic/claude-3.5-sonnet")
            provider = OpenRouterProvider(config)

            with pytest.raises(AuthenticationError):
                await provider.initialize()


# ============================================================================
# Cohere Provider Tests
# ============================================================================

class TestCohereProvider:
    """Test Cohere provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        with patch.dict('sys.modules', {'cohere': MagicMock()}):
            from sentimatrix.providers.llm.cohere_provider import CohereProvider

            config = LLMConfig(
                provider="cohere",
                model="command-r-plus",
                api_key="test_key",
            )
            provider = CohereProvider(config)

            assert provider._model == "command-r-plus"
            assert provider._api_key == "test_key"

    def test_provider_info(self):
        """Test provider information."""
        with patch.dict('sys.modules', {'cohere': MagicMock()}):
            from sentimatrix.providers.llm.cohere_provider import CohereProvider

            config = LLMConfig(provider="cohere", model="command-r-plus")
            provider = CohereProvider(config)

            info = provider.info
            assert info.name == "cohere"
            assert info.provider_type == ProviderType.LLM
            assert info.capabilities.streaming is True
            assert info.capabilities.embeddings is True

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        mock_cohere = MagicMock()
        mock_httpx = MagicMock()

        with patch.dict('sys.modules', {'cohere': mock_cohere, 'httpx': mock_httpx}):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop('COHERE_API_KEY', None)

                from sentimatrix.providers.llm.cohere_provider import CohereProvider

                config = LLMConfig(provider="cohere", model="command-r-plus")
                provider = CohereProvider(config)

                with pytest.raises(AuthenticationError):
                    await provider.initialize()


# ============================================================================
# LM Studio Provider Tests
# ============================================================================

class TestLMStudioProvider:
    """Test LM Studio provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.lmstudio_provider import LMStudioProvider

        config = LLMConfig(
            provider="lmstudio",
            model="local-model",
        )
        provider = LMStudioProvider(config)

        assert provider._model == "local-model"
        assert "localhost:1234" in provider._base_url

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.lmstudio_provider import LMStudioProvider

        config = LLMConfig(provider="lmstudio", model="local-model")
        provider = LMStudioProvider(config)

        info = provider.info
        assert info.name == "lmstudio"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert "GGUF" in info.description

    @pytest.mark.asyncio
    async def test_initialize_no_api_key_required(self):
        """Test initialization succeeds without API key (local provider)."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.lmstudio_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.lmstudio_provider import LMStudioProvider

            config = LLMConfig(provider="lmstudio", model="local-model")
            provider = LMStudioProvider(config)

            await provider.initialize()
            assert provider._initialized


# ============================================================================
# vLLM Provider Tests
# ============================================================================

class TestVLLMProvider:
    """Test vLLM provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.vllm_provider import VLLMProvider

        config = LLMConfig(
            provider="vllm",
            model="meta-llama/Llama-2-7b-chat-hf",
        )
        provider = VLLMProvider(config)

        assert "Llama-2" in provider._model
        assert "localhost:8000" in provider._base_url

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.vllm_provider import VLLMProvider

        config = LLMConfig(provider="vllm", model="local-model")
        provider = VLLMProvider(config)

        info = provider.info
        assert info.name == "vllm"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert "PagedAttention" in info.description

    @pytest.mark.asyncio
    async def test_initialize_no_api_key_required(self):
        """Test initialization succeeds without API key (local provider)."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.vllm_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.vllm_provider import VLLMProvider

            config = LLMConfig(provider="vllm", model="local-model")
            provider = VLLMProvider(config)

            await provider.initialize()
            assert provider._initialized


# ============================================================================
# DeepSeek Provider Tests
# ============================================================================

class TestDeepSeekProvider:
    """Test DeepSeek provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.deepseek_provider import DeepSeekProvider

        config = LLMConfig(
            provider="deepseek",
            model="deepseek-chat",
            api_key="test_key",
        )
        provider = DeepSeekProvider(config)

        assert provider._model == "deepseek-chat"
        assert provider._api_key == "test_key"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.deepseek_provider import DeepSeekProvider

        config = LLMConfig(provider="deepseek", model="deepseek-chat")
        provider = DeepSeekProvider(config)

        info = provider.info
        assert info.name == "deepseek"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert "reasoning" in info.description.lower()

    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('DEEPSEEK_API_KEY', None)

            from sentimatrix.providers.llm.deepseek_provider import DeepSeekProvider

            config = LLMConfig(provider="deepseek", model="deepseek-chat")
            provider = DeepSeekProvider(config)

            with pytest.raises(AuthenticationError):
                await provider.initialize()

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generate method."""
        mock_response = MockHTTPResponse(create_mock_chat_response("Hello from DeepSeek!", "deepseek-chat"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.deepseek_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.deepseek_provider import DeepSeekProvider

            config = LLMConfig(provider="deepseek", model="deepseek-chat", api_key="test_key")
            provider = DeepSeekProvider(config)
            await provider.initialize()

            response = await provider.generate("Hello!")

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello from DeepSeek!"
            assert response.provider == "deepseek"


# ============================================================================
# Provider Registration Tests
# ============================================================================

class TestProviderRegistration:
    """Test that all providers are properly registered."""

    def test_all_providers_registered(self):
        """Test all new providers are registered."""
        from sentimatrix.providers.base import get_provider, ProviderType

        # Cloud Enterprise
        assert get_provider("azure-openai", ProviderType.LLM) is not None
        assert get_provider("bedrock", ProviderType.LLM) is not None

        # Fast Inference
        assert get_provider("cerebras", ProviderType.LLM) is not None
        assert get_provider("fireworks", ProviderType.LLM) is not None
        assert get_provider("together", ProviderType.LLM) is not None

        # Router
        assert get_provider("openrouter", ProviderType.LLM) is not None

        # Specialized
        assert get_provider("mistral", ProviderType.LLM) is not None
        assert get_provider("cohere", ProviderType.LLM) is not None
        assert get_provider("deepseek", ProviderType.LLM) is not None

        # Local
        assert get_provider("lmstudio", ProviderType.LLM) is not None
        assert get_provider("vllm", ProviderType.LLM) is not None

    def test_all_providers_importable(self):
        """Test all providers can be imported from __init__."""
        from sentimatrix.providers.llm import (
            AzureOpenAIProvider,
            BedrockProvider,
            MistralProvider,
            CerebrasProvider,
            FireworksProvider,
            TogetherProvider,
            OpenRouterProvider,
            CohereProvider,
            LMStudioProvider,
            VLLMProvider,
            DeepSeekProvider,
        )

        assert AzureOpenAIProvider is not None
        assert BedrockProvider is not None
        assert MistralProvider is not None
        assert CerebrasProvider is not None
        assert FireworksProvider is not None
        assert TogetherProvider is not None
        assert OpenRouterProvider is not None
        assert CohereProvider is not None
        assert LMStudioProvider is not None
        assert VLLMProvider is not None
        assert DeepSeekProvider is not None
