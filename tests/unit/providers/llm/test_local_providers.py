"""
Unit tests for Local Inference LLM Providers.

Tests cover:
- llama.cpp (LlamaCppProvider)
- text-generation-webui (TextGenProvider)
- ExLlamaV2 (ExLlamaV2Provider)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any
import os

from sentimatrix.core.config import LLMConfig
from sentimatrix.providers.base import LLMResponse, ProviderType, TokenUsage


class MockHTTPResponse:
    """Mock httpx response."""
    def __init__(self, data: dict, status_code: int = 200):
        self._data = data
        self.status_code = status_code
        self.text = str(data)

    def json(self):
        return self._data


def create_mock_chat_response(content: str = "Hello!", model: str = "local-model"):
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
# llama.cpp Provider Tests
# ============================================================================

class TestLlamaCppProvider:
    """Test llama.cpp provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.llamacpp_provider import LlamaCppProvider

        config = LLMConfig(
            provider="llamacpp",
            model="local-model",
        )
        provider = LlamaCppProvider(config)

        assert provider._model == "local-model"
        assert "localhost:8080" in provider._base_url

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.llamacpp_provider import LlamaCppProvider

        config = LLMConfig(provider="llamacpp", model="local-model")
        provider = LlamaCppProvider(config)

        info = provider.info
        assert info.name == "llamacpp"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert info.capabilities.embeddings is True
        assert "GGUF" in info.description

    @pytest.mark.asyncio
    async def test_initialize_no_api_key_required(self):
        """Test initialization succeeds without API key (local provider)."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.llamacpp_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.llamacpp_provider import LlamaCppProvider

            config = LLMConfig(provider="llamacpp", model="local-model")
            provider = LlamaCppProvider(config)

            await provider.initialize()
            assert provider._initialized

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generate method."""
        mock_response = MockHTTPResponse(create_mock_chat_response("Hello from llama.cpp!", "llama-7b"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.llamacpp_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.llamacpp_provider import LlamaCppProvider

            config = LLMConfig(provider="llamacpp", model="local-model")
            provider = LlamaCppProvider(config)
            await provider.initialize()

            response = await provider.generate("Hello!")

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello from llama.cpp!"
            assert response.provider == "llamacpp"

    @pytest.mark.asyncio
    async def test_complete_endpoint(self):
        """Test raw text completion endpoint."""
        mock_response = MockHTTPResponse({
            "content": "Completed text",
            "tokens_evaluated": 5,
            "tokens_predicted": 15,
            "stop_type": "stop",
        })
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.llamacpp_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.llamacpp_provider import LlamaCppProvider

            config = LLMConfig(provider="llamacpp", model="local-model")
            provider = LlamaCppProvider(config)
            await provider.initialize()

            response = await provider.complete("Complete this:")

            assert response.content == "Completed text"
            assert response.usage.prompt_tokens == 5
            assert response.usage.completion_tokens == 15


# ============================================================================
# text-generation-webui Provider Tests
# ============================================================================

class TestTextGenProvider:
    """Test text-generation-webui provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.textgen_provider import TextGenProvider

        config = LLMConfig(
            provider="textgen",
            model="local-model",
        )
        provider = TextGenProvider(config)

        assert provider._model == "local-model"
        assert "localhost:5000" in provider._base_url

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.textgen_provider import TextGenProvider

        config = LLMConfig(provider="textgen", model="local-model")
        provider = TextGenProvider(config)

        info = provider.info
        assert info.name == "textgen"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert "Multi-backend" in info.description

    @pytest.mark.asyncio
    async def test_initialize_no_api_key_required(self):
        """Test initialization succeeds without API key (local provider)."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.textgen_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.textgen_provider import TextGenProvider

            config = LLMConfig(provider="textgen", model="local-model")
            provider = TextGenProvider(config)

            await provider.initialize()
            assert provider._initialized

    @pytest.mark.asyncio
    async def test_generate_openai_mode(self):
        """Test generate method in OpenAI-compatible mode."""
        mock_response = MockHTTPResponse(create_mock_chat_response("Hello from textgen!", "vicuna-13b"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.textgen_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.textgen_provider import TextGenProvider

            config = LLMConfig(provider="textgen", model="local-model")
            provider = TextGenProvider(config)
            await provider.initialize()

            response = await provider.generate("Hello!")

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello from textgen!"
            assert response.provider == "textgen"


# ============================================================================
# ExLlamaV2 Provider Tests
# ============================================================================

class TestExLlamaV2Provider:
    """Test ExLlamaV2 provider."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.llm.exllamav2_provider import ExLlamaV2Provider

        config = LLMConfig(
            provider="exllamav2",
            model="local-model",
        )
        provider = ExLlamaV2Provider(config)

        assert provider._model == "local-model"
        assert "localhost:5000" in provider._base_url

    def test_init_with_api_key(self):
        """Test initialization with optional API key for TabbyAPI auth."""
        from sentimatrix.providers.llm.exllamav2_provider import ExLlamaV2Provider

        config = LLMConfig(
            provider="exllamav2",
            model="local-model",
            api_key="tabby-api-key",
        )
        provider = ExLlamaV2Provider(config)

        assert provider._api_key == "tabby-api-key"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.llm.exllamav2_provider import ExLlamaV2Provider

        config = LLMConfig(provider="exllamav2", model="local-model")
        provider = ExLlamaV2Provider(config)

        info = provider.info
        assert info.name == "exllamav2"
        assert info.provider_type == ProviderType.LLM
        assert info.capabilities.streaming is True
        assert "GPTQ" in info.description or "EXL2" in info.description

    @pytest.mark.asyncio
    async def test_initialize_no_api_key_required(self):
        """Test initialization succeeds without API key (local provider)."""
        mock_httpx = MagicMock()
        mock_client = AsyncMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.exllamav2_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.exllamav2_provider import ExLlamaV2Provider

            config = LLMConfig(provider="exllamav2", model="local-model")
            provider = ExLlamaV2Provider(config)

            await provider.initialize()
            assert provider._initialized

    @pytest.mark.asyncio
    async def test_generate(self):
        """Test generate method."""
        mock_response = MockHTTPResponse(create_mock_chat_response("Hello from ExLlamaV2!", "llama-2-13b-exl2"))
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)

        with patch.dict('sys.modules', {'httpx': mock_httpx}):
            import sentimatrix.providers.llm.exllamav2_provider as module
            module._httpx = None

            from sentimatrix.providers.llm.exllamav2_provider import ExLlamaV2Provider

            config = LLMConfig(provider="exllamav2", model="local-model")
            provider = ExLlamaV2Provider(config)
            await provider.initialize()

            response = await provider.generate("Hello!")

            assert isinstance(response, LLMResponse)
            assert response.content == "Hello from ExLlamaV2!"
            assert response.provider == "exllamav2"


# ============================================================================
# Provider Registration Tests
# ============================================================================

class TestLocalProviderRegistration:
    """Test that all local providers are properly registered."""

    def test_all_local_providers_registered(self):
        """Test all local providers are registered."""
        from sentimatrix.providers.base import get_provider, ProviderType

        # Local providers
        assert get_provider("ollama", ProviderType.LLM) is not None
        assert get_provider("lmstudio", ProviderType.LLM) is not None
        assert get_provider("vllm", ProviderType.LLM) is not None
        assert get_provider("llamacpp", ProviderType.LLM) is not None
        assert get_provider("textgen", ProviderType.LLM) is not None
        assert get_provider("exllamav2", ProviderType.LLM) is not None

    def test_all_local_providers_importable(self):
        """Test all local providers can be imported from __init__."""
        from sentimatrix.providers.llm import (
            OllamaProvider,
            LMStudioProvider,
            VLLMProvider,
            LlamaCppProvider,
            TextGenProvider,
            ExLlamaV2Provider,
        )

        assert OllamaProvider is not None
        assert LMStudioProvider is not None
        assert VLLMProvider is not None
        assert LlamaCppProvider is not None
        assert TextGenProvider is not None
        assert ExLlamaV2Provider is not None


# ============================================================================
# Config Enum Tests
# ============================================================================

class TestLocalProviderConfig:
    """Test that local providers are in the config enum."""

    def test_local_providers_in_enum(self):
        """Test all local providers are in LLMProvider enum."""
        from sentimatrix.core.config import LLMProvider

        # Check all local providers exist in enum
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.LMSTUDIO.value == "lmstudio"
        assert LLMProvider.VLLM.value == "vllm"
        assert LLMProvider.LLAMACPP.value == "llamacpp"
        assert LLMProvider.TEXTGEN.value == "textgen"
        assert LLMProvider.EXLLAMAV2.value == "exllamav2"
