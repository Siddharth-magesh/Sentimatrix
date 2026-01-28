"""
Unit Tests for HuggingFace Model Provider

Tests the HuggingFace model provider including:
- Device detection
- Model loading
- Single and batch prediction
- Error handling
- Caching
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from sentimatrix.core.config import ModelConfig
from sentimatrix.core.exceptions import (
    DeviceError,
    ModelInferenceError,
    ModelLoadError,
    ModelNotFoundError,
    ProviderInitializationError,
)
from sentimatrix.providers.base import PredictionResult, ProviderType


class TestDeviceDetection:
    """Tests for device detection functionality."""

    def test_detect_device_cpu(self):
        """Test CPU device detection."""
        from sentimatrix.providers.models.huggingface import _detect_device

        # CPU should always be available
        device = _detect_device("cpu")
        assert device == "cpu"

    def test_detect_device_auto(self):
        """Test auto device detection."""
        from sentimatrix.providers.models.huggingface import _detect_device

        # Auto should return a valid device
        device = _detect_device("auto")
        assert device in ("cpu", "cuda", "mps")

    def test_detect_device_cuda_check(self):
        """Test CUDA device detection logic."""
        from sentimatrix.providers.models.huggingface import _detect_device

        try:
            import torch
            if torch.cuda.is_available():
                device = _detect_device("cuda")
                assert device == "cuda"
            else:
                # If CUDA not available, should raise error
                with pytest.raises(DeviceError):
                    _detect_device("cuda")
        except ImportError:
            # Without torch, should raise error for cuda
            with pytest.raises(DeviceError):
                _detect_device("cuda")

    def test_detect_device_returns_valid_type(self):
        """Test that detect_device returns valid device type."""
        from sentimatrix.providers.models.huggingface import _detect_device

        for device_type in ["cpu", "auto"]:
            result = _detect_device(device_type)
            assert result in ("cpu", "cuda", "mps")

    def test_get_device_info_returns_dict(self):
        """Test getting device information returns a dictionary."""
        from sentimatrix.providers.models.huggingface import _get_device_info

        info = _get_device_info()

        assert isinstance(info, dict)
        assert "cpu" in info
        assert info["cpu"] is True
        assert "cuda" in info
        assert "mps" in info
        assert "cuda_device_count" in info


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_model_info_creation(self):
        """Test creating ModelInfo."""
        from sentimatrix.providers.models.huggingface import ModelInfo, ModelType

        info = ModelInfo(
            name="test-model",
            model_type=ModelType.SENTIMENT,
            device="cpu",
            num_labels=3,
            labels=["negative", "neutral", "positive"],
            max_length=512,
        )

        assert info.name == "test-model"
        assert info.model_type == ModelType.SENTIMENT
        assert info.num_labels == 3
        assert len(info.labels) == 3

    def test_model_info_to_dict(self):
        """Test ModelInfo serialization."""
        from sentimatrix.providers.models.huggingface import ModelInfo, ModelType

        info = ModelInfo(
            name="test-model",
            model_type=ModelType.EMOTION,
            device="cuda",
            num_labels=28,
            labels=["joy", "sadness"] + [f"emotion_{i}" for i in range(26)],
            max_length=256,
        )

        info_dict = info.to_dict()

        assert info_dict["name"] == "test-model"
        assert info_dict["model_type"] == "emotion"
        assert info_dict["device"] == "cuda"
        assert info_dict["num_labels"] == 28


class TestHuggingFaceModelProvider:
    """Tests for HuggingFaceModelProvider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration for testing."""
        return ModelConfig(
            sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            emotion_model="SamLowe/roberta-base-go_emotions",
            device="cpu",
            batch_size=16,
            max_length=512,
            cache_models=False,  # Disable caching in tests
        )

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        mock_model = MagicMock()
        mock_model.config.num_labels = 3
        mock_model.config.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        mock_tokenizer = MagicMock()
        mock_tokenizer.model_max_length = 512
        mock_tokenizer.return_value = {
            "input_ids": MagicMock(),
            "attention_mask": MagicMock(),
        }

        return mock_model, mock_tokenizer

    def test_provider_init(self, model_config):
        """Test provider initialization."""
        from sentimatrix.providers.models.huggingface import (
            HuggingFaceModelProvider,
            ModelType,
        )

        provider = HuggingFaceModelProvider(
            config=model_config,
            model_type=ModelType.SENTIMENT,
        )

        assert provider._model_name == model_config.sentiment_model
        assert provider._model_type == ModelType.SENTIMENT
        assert not provider._initialized

    def test_provider_info(self, model_config):
        """Test provider info property."""
        from sentimatrix.providers.models.huggingface import HuggingFaceModelProvider

        provider = HuggingFaceModelProvider(config=model_config)
        info = provider.info

        assert info.name == "huggingface"
        assert info.provider_type == ProviderType.MODEL
        assert info.capabilities.batch_processing is True
        assert info.capabilities.gpu_support is True

    @pytest.mark.asyncio
    async def test_initialize_not_initialized(self, model_config):
        """Test that provider is not initialized by default."""
        from sentimatrix.providers.models.huggingface import HuggingFaceModelProvider

        provider = HuggingFaceModelProvider(config=model_config)

        assert not provider.is_initialized

    @pytest.mark.asyncio
    async def test_predict_without_init_raises_error(self, model_config):
        """Test that predict raises error when not initialized."""
        from sentimatrix.providers.models.huggingface import HuggingFaceModelProvider

        provider = HuggingFaceModelProvider(config=model_config)

        with pytest.raises(ProviderInitializationError):
            await provider.predict("Test text")

    @pytest.mark.asyncio
    async def test_predict_batch_without_init_raises_error(self, model_config):
        """Test that predict_batch raises error when not initialized."""
        from sentimatrix.providers.models.huggingface import HuggingFaceModelProvider

        provider = HuggingFaceModelProvider(config=model_config)

        with pytest.raises(ProviderInitializationError):
            await provider.predict_batch(["Test text"])

    def test_get_model_info_not_loaded(self, model_config):
        """Test get_model_info when model not loaded."""
        from sentimatrix.providers.models.huggingface import HuggingFaceModelProvider

        provider = HuggingFaceModelProvider(config=model_config)
        info = provider.get_model_info()

        assert info["loaded"] is False
        assert info["name"] == model_config.sentiment_model


class TestSentimentModelProvider:
    """Tests for SentimentModelProvider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device="cpu",
            cache_models=False,
        )

    def test_sentiment_provider_init(self, model_config):
        """Test sentiment provider initialization."""
        from sentimatrix.providers.models.huggingface import SentimentModelProvider

        provider = SentimentModelProvider(config=model_config)

        assert provider._model_name == model_config.sentiment_model
        assert "sentiment" in provider.info.name

    def test_sentiment_provider_info(self, model_config):
        """Test sentiment provider info."""
        from sentimatrix.providers.models.huggingface import SentimentModelProvider

        provider = SentimentModelProvider(config=model_config)
        info = provider.info

        assert "sentiment" in info.description.lower()
        assert any("sentiment" in m.lower() for m in info.supported_models)


class TestEmotionModelProvider:
    """Tests for EmotionModelProvider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            emotion_model="SamLowe/roberta-base-go_emotions",
            device="cpu",
            cache_models=False,
        )

    def test_emotion_provider_init(self, model_config):
        """Test emotion provider initialization."""
        from sentimatrix.providers.models.huggingface import EmotionModelProvider

        provider = EmotionModelProvider(config=model_config)

        assert provider._model_name == model_config.emotion_model
        assert "emotion" in provider.info.name

    def test_emotion_provider_info(self, model_config):
        """Test emotion provider info."""
        from sentimatrix.providers.models.huggingface import EmotionModelProvider

        provider = EmotionModelProvider(config=model_config)
        info = provider.info

        assert "emotion" in info.description.lower()
        assert any("emotion" in m.lower() for m in info.supported_models)


class TestModelCache:
    """Tests for model caching functionality."""

    def test_clear_model_cache(self):
        """Test clearing model cache."""
        from sentimatrix.providers.models.huggingface import (
            _model_cache,
            clear_model_cache,
            get_cached_models,
        )

        # Add something to cache
        _model_cache["test:cpu"] = (MagicMock(), MagicMock(), MagicMock())

        assert len(get_cached_models()) > 0

        clear_model_cache()

        assert len(get_cached_models()) == 0

    def test_get_cached_models(self):
        """Test getting list of cached models."""
        from sentimatrix.providers.models.huggingface import (
            _model_cache,
            clear_model_cache,
            get_cached_models,
        )

        clear_model_cache()

        _model_cache["model1:cpu"] = (MagicMock(), MagicMock(), MagicMock())
        _model_cache["model2:cuda"] = (MagicMock(), MagicMock(), MagicMock())

        cached = get_cached_models()

        assert len(cached) == 2
        assert "model1:cpu" in cached
        assert "model2:cuda" in cached

        clear_model_cache()


class TestProviderRegistration:
    """Tests for provider registration."""

    def test_providers_registered(self):
        """Test that providers are registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "huggingface" in providers["model"]
        assert "huggingface-sentiment" in providers["model"]
        assert "huggingface-emotion" in providers["model"]

    def test_get_huggingface_provider(self):
        """Test getting HuggingFace provider from registry."""
        from sentimatrix.providers.base import get_provider

        from sentimatrix.providers.models.huggingface import HuggingFaceModelProvider

        config = ModelConfig(device="cpu", cache_models=False)
        provider = get_provider("huggingface", "model", config)

        assert isinstance(provider, HuggingFaceModelProvider)
