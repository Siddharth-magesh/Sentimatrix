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

    def test_extended_providers_registered(self):
        """Test that extended providers are registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        # Check extended sentiment providers
        assert "distilbert-sentiment" in providers["model"]
        assert "siebert-sentiment" in providers["model"]
        assert "twitter-sentiment" in providers["model"]
        assert "multilingual-sentiment" in providers["model"]

        # Check extended emotion providers
        assert "twitter-emotion" in providers["model"]
        assert "t5-emotion" in providers["model"]

        # Check ABSA providers
        assert "deberta-absa" in providers["model"]
        assert "instruct-absa" in providers["model"]


class TestDistilBertSentimentProvider:
    """Tests for DistilBERT sentiment provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_distilbert_provider_init(self, model_config):
        """Test DistilBERT provider initialization."""
        from sentimatrix.providers.models.huggingface import DistilBertSentimentProvider

        provider = DistilBertSentimentProvider(config=model_config)

        assert "distilbert" in provider._model_name.lower()
        assert "sst-2" in provider._model_name.lower()

    def test_distilbert_provider_info(self, model_config):
        """Test DistilBERT provider info."""
        from sentimatrix.providers.models.huggingface import DistilBertSentimentProvider

        provider = DistilBertSentimentProvider(config=model_config)
        info = provider.info

        assert info.name == "distilbert-sentiment"
        assert "fast" in info.description.lower() or "binary" in info.description.lower()

    def test_distilbert_provider_default_model(self, model_config):
        """Test DistilBERT provider uses default model."""
        from sentimatrix.providers.models.huggingface import DistilBertSentimentProvider

        provider = DistilBertSentimentProvider(config=model_config)

        assert provider._model_name == DistilBertSentimentProvider.DEFAULT_MODEL


class TestSiebertSentimentProvider:
    """Tests for SiEBERT sentiment provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_siebert_provider_init(self, model_config):
        """Test SiEBERT provider initialization."""
        from sentimatrix.providers.models.huggingface import SiebertSentimentProvider

        provider = SiebertSentimentProvider(config=model_config)

        assert "siebert" in provider._model_name.lower()
        assert "roberta-large" in provider._model_name.lower()

    def test_siebert_provider_info(self, model_config):
        """Test SiEBERT provider info."""
        from sentimatrix.providers.models.huggingface import SiebertSentimentProvider

        provider = SiebertSentimentProvider(config=model_config)
        info = provider.info

        assert info.name == "siebert-sentiment"
        assert "high-accuracy" in info.description.lower() or "roberta" in info.description.lower()


class TestTwitterSentimentProvider:
    """Tests for Twitter sentiment provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_twitter_sentiment_provider_init(self, model_config):
        """Test Twitter sentiment provider initialization."""
        from sentimatrix.providers.models.huggingface import TwitterSentimentProvider

        provider = TwitterSentimentProvider(config=model_config)

        assert "twitter" in provider._model_name.lower()
        assert "sentiment" in provider._model_name.lower()

    def test_twitter_sentiment_provider_info(self, model_config):
        """Test Twitter sentiment provider info."""
        from sentimatrix.providers.models.huggingface import TwitterSentimentProvider

        provider = TwitterSentimentProvider(config=model_config)
        info = provider.info

        assert info.name == "twitter-sentiment"
        assert "twitter" in info.description.lower()

    def test_twitter_text_preprocessing(self):
        """Test Twitter text preprocessing."""
        from sentimatrix.providers.models.huggingface import TwitterSentimentProvider

        # Test username replacement
        text1 = "@john_doe said hello"
        preprocessed1 = TwitterSentimentProvider.preprocess_twitter_text(text1)
        assert "@user" in preprocessed1
        assert "@john_doe" not in preprocessed1

        # Test URL replacement
        text2 = "Check this https://example.com/page"
        preprocessed2 = TwitterSentimentProvider.preprocess_twitter_text(text2)
        assert "http" in preprocessed2
        assert "example.com" not in preprocessed2

        # Test combined
        text3 = "@user1 visit http://test.com for more"
        preprocessed3 = TwitterSentimentProvider.preprocess_twitter_text(text3)
        assert "@user" in preprocessed3
        assert "http" in preprocessed3


class TestMultilingualSentimentProvider:
    """Tests for multilingual sentiment provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_multilingual_provider_init(self, model_config):
        """Test multilingual provider initialization."""
        from sentimatrix.providers.models.huggingface import MultilingualSentimentProvider

        provider = MultilingualSentimentProvider(config=model_config)

        assert "multilingual" in provider._model_name.lower()

    def test_multilingual_provider_info(self, model_config):
        """Test multilingual provider info."""
        from sentimatrix.providers.models.huggingface import MultilingualSentimentProvider

        provider = MultilingualSentimentProvider(config=model_config)
        info = provider.info

        assert info.name == "multilingual-sentiment"
        assert "multilingual" in info.description.lower()


class TestTwitterEmotionProvider:
    """Tests for Twitter emotion provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_twitter_emotion_provider_init(self, model_config):
        """Test Twitter emotion provider initialization."""
        from sentimatrix.providers.models.huggingface import TwitterEmotionProvider

        provider = TwitterEmotionProvider(config=model_config)

        assert "twitter" in provider._model_name.lower()
        assert "emotion" in provider._model_name.lower()

    def test_twitter_emotion_provider_info(self, model_config):
        """Test Twitter emotion provider info."""
        from sentimatrix.providers.models.huggingface import TwitterEmotionProvider

        provider = TwitterEmotionProvider(config=model_config)
        info = provider.info

        assert info.name == "twitter-emotion"
        assert "emotion" in info.description.lower()

    def test_twitter_emotion_text_preprocessing(self):
        """Test Twitter emotion text preprocessing."""
        from sentimatrix.providers.models.huggingface import TwitterEmotionProvider

        text = "@someone I'm so happy! http://link.com"
        preprocessed = TwitterEmotionProvider.preprocess_twitter_text(text)

        assert "@user" in preprocessed
        assert "happy" in preprocessed


class TestT5EmotionProvider:
    """Tests for T5 emotion provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_t5_emotion_provider_init(self, model_config):
        """Test T5 emotion provider initialization."""
        from sentimatrix.providers.models.huggingface import T5EmotionProvider

        provider = T5EmotionProvider(config=model_config)

        assert "t5" in provider._model_name.lower()
        assert "emotion" in provider._model_name.lower()

    def test_t5_emotion_provider_info(self, model_config):
        """Test T5 emotion provider info."""
        from sentimatrix.providers.models.huggingface import T5EmotionProvider

        provider = T5EmotionProvider(config=model_config)
        info = provider.info

        assert info.name == "t5-emotion"
        # T5 doesn't batch well
        assert info.capabilities.batch_processing is False


class TestDeBERTaABSAProvider:
    """Tests for DeBERTa ABSA provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_deberta_absa_provider_init(self, model_config):
        """Test DeBERTa ABSA provider initialization."""
        from sentimatrix.providers.models.huggingface import DeBERTaABSAProvider

        provider = DeBERTaABSAProvider(config=model_config)

        assert "deberta" in provider._model_name.lower()
        assert "absa" in provider._model_name.lower()

    def test_deberta_absa_provider_large_model(self, model_config):
        """Test DeBERTa ABSA provider with large model."""
        from sentimatrix.providers.models.huggingface import DeBERTaABSAProvider

        provider = DeBERTaABSAProvider(config=model_config, use_large=True)

        assert "large" in provider._model_name.lower()

    def test_deberta_absa_provider_info(self, model_config):
        """Test DeBERTa ABSA provider info."""
        from sentimatrix.providers.models.huggingface import DeBERTaABSAProvider

        provider = DeBERTaABSAProvider(config=model_config)
        info = provider.info

        assert info.name == "deberta-absa"
        assert "aspect" in info.description.lower()

    def test_deberta_absa_labels(self):
        """Test DeBERTa ABSA labels."""
        from sentimatrix.providers.models.huggingface import DeBERTaABSAProvider

        assert "Negative" in DeBERTaABSAProvider.LABELS
        assert "Neutral" in DeBERTaABSAProvider.LABELS
        assert "Positive" in DeBERTaABSAProvider.LABELS


class TestInstructABSAProvider:
    """Tests for InstructABSA provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_instruct_absa_provider_init(self, model_config):
        """Test InstructABSA provider initialization."""
        from sentimatrix.providers.models.huggingface import InstructABSAProvider

        provider = InstructABSAProvider(config=model_config)

        assert "instruct" in provider._model_name.lower()
        assert "joint" in provider._model_name.lower()

    def test_instruct_absa_provider_info(self, model_config):
        """Test InstructABSA provider info."""
        from sentimatrix.providers.models.huggingface import InstructABSAProvider

        provider = InstructABSAProvider(config=model_config)
        info = provider.info

        assert info.name == "instruct-absa"
        assert "instruction" in info.description.lower() or "joint" in info.description.lower()
        # T5-based doesn't batch well
        assert info.capabilities.batch_processing is False


class TestABSAResult:
    """Tests for ABSAResult dataclass."""

    def test_absa_result_creation(self):
        """Test creating ABSAResult."""
        from sentimatrix.providers.models.huggingface import ABSAResult

        result = ABSAResult(
            text="The food was great",
            aspect="food",
            sentiment="Positive",
            confidence=0.95,
            all_scores={"Negative": 0.02, "Neutral": 0.03, "Positive": 0.95},
            model_name="test-model",
        )

        assert result.text == "The food was great"
        assert result.aspect == "food"
        assert result.sentiment == "Positive"
        assert result.confidence == 0.95

    def test_absa_result_to_dict(self):
        """Test ABSAResult serialization."""
        from sentimatrix.providers.models.huggingface import ABSAResult

        result = ABSAResult(
            text="Service was slow",
            aspect="service",
            sentiment="Negative",
            confidence=0.88,
            model_name="deberta-absa",
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Service was slow"
        assert result_dict["aspect"] == "service"
        assert result_dict["sentiment"] == "Negative"
        assert result_dict["confidence"] == 0.88
        assert result_dict["model_name"] == "deberta-absa"


class TestModelHelperFunctions:
    """Tests for model helper functions."""

    def test_get_available_sentiment_models(self):
        """Test getting available sentiment models."""
        from sentimatrix.providers.models.huggingface import get_available_sentiment_models

        models = get_available_sentiment_models()

        assert isinstance(models, dict)
        assert len(models) > 0
        assert "twitter-roberta-latest" in models
        assert "distilbert-sst2" in models
        assert "siebert-large" in models

    def test_get_available_emotion_models(self):
        """Test getting available emotion models."""
        from sentimatrix.providers.models.huggingface import get_available_emotion_models

        models = get_available_emotion_models()

        assert isinstance(models, dict)
        assert len(models) > 0
        assert "go-emotions" in models
        assert "twitter-emotion" in models
        assert "t5-emotion" in models

    def test_get_available_absa_models(self):
        """Test getting available ABSA models."""
        from sentimatrix.providers.models.huggingface import get_available_absa_models

        models = get_available_absa_models()

        assert isinstance(models, dict)
        assert len(models) > 0
        assert "deberta-absa-base" in models
        assert "deberta-absa-large" in models
        assert "instruct-absa" in models

    def test_get_available_embedding_models(self):
        """Test getting available embedding models."""
        from sentimatrix.providers.models.huggingface import get_available_embedding_models

        models = get_available_embedding_models()

        assert isinstance(models, dict)
        assert len(models) > 0
        assert "minilm" in models
        assert "mpnet" in models

    def test_get_available_translation_models(self):
        """Test getting available translation models."""
        from sentimatrix.providers.models.huggingface import get_available_translation_models

        models = get_available_translation_models()

        assert isinstance(models, dict)
        assert len(models) > 0
        assert "en-de" in models
        assert "en-fr" in models

    def test_get_available_whisper_models(self):
        """Test getting available Whisper models."""
        from sentimatrix.providers.models.huggingface import get_available_whisper_models

        models = get_available_whisper_models()

        assert isinstance(models, dict)
        assert len(models) > 0
        assert "base" in models
        assert "large" in models


# ============================================================================
# XLM-RoBERTa Sentiment Provider Tests
# ============================================================================


class TestXLMRobertaSentimentProvider:
    """Tests for XLM-RoBERTa multilingual sentiment provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_xlm_roberta_provider_init(self, model_config):
        """Test XLM-RoBERTa provider initialization."""
        from sentimatrix.providers.models.huggingface import XLMRobertaSentimentProvider

        provider = XLMRobertaSentimentProvider(config=model_config)

        assert "xlm-roberta" in provider._model_name.lower()
        assert "sentiment" in provider._model_name.lower()

    def test_xlm_roberta_provider_info(self, model_config):
        """Test XLM-RoBERTa provider info."""
        from sentimatrix.providers.models.huggingface import XLMRobertaSentimentProvider

        provider = XLMRobertaSentimentProvider(config=model_config)
        info = provider.info

        assert info.name == "xlm-roberta-sentiment"
        assert "multilingual" in info.description.lower()

    def test_xlm_roberta_default_model(self, model_config):
        """Test XLM-RoBERTa uses correct default model."""
        from sentimatrix.providers.models.huggingface import XLMRobertaSentimentProvider

        provider = XLMRobertaSentimentProvider(config=model_config)

        assert provider._model_name == XLMRobertaSentimentProvider.DEFAULT_MODEL

    def test_xlm_roberta_registration(self):
        """Test XLM-RoBERTa provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "xlm-roberta-sentiment" in providers["model"]


# ============================================================================
# FinBERT Provider Tests
# ============================================================================


class TestFinBERTProvider:
    """Tests for FinBERT financial sentiment provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_finbert_provider_init(self, model_config):
        """Test FinBERT provider initialization."""
        from sentimatrix.providers.models.huggingface import FinBERTProvider

        provider = FinBERTProvider(config=model_config)

        assert "finbert" in provider._model_name.lower()

    def test_finbert_provider_info(self, model_config):
        """Test FinBERT provider info."""
        from sentimatrix.providers.models.huggingface import FinBERTProvider

        provider = FinBERTProvider(config=model_config)
        info = provider.info

        assert info.name == "finbert"
        assert "financial" in info.description.lower()

    def test_finbert_default_model(self, model_config):
        """Test FinBERT uses correct default model."""
        from sentimatrix.providers.models.huggingface import FinBERTProvider

        provider = FinBERTProvider(config=model_config)

        assert provider._model_name == FinBERTProvider.DEFAULT_MODEL
        assert "ProsusAI" in provider._model_name

    def test_finbert_registration(self):
        """Test FinBERT provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "finbert" in providers["model"]


class TestFinBERTToneProvider:
    """Tests for FinBERT-Tone financial tone provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_finbert_tone_provider_init(self, model_config):
        """Test FinBERT-Tone provider initialization."""
        from sentimatrix.providers.models.huggingface import FinBERTToneProvider

        provider = FinBERTToneProvider(config=model_config)

        assert "finbert" in provider._model_name.lower()
        assert "tone" in provider._model_name.lower()

    def test_finbert_tone_provider_info(self, model_config):
        """Test FinBERT-Tone provider info."""
        from sentimatrix.providers.models.huggingface import FinBERTToneProvider

        provider = FinBERTToneProvider(config=model_config)
        info = provider.info

        assert info.name == "finbert-tone"
        assert "tone" in info.description.lower()

    def test_finbert_tone_registration(self):
        """Test FinBERT-Tone provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "finbert-tone" in providers["model"]


# ============================================================================
# Zero-Shot Classification Provider Tests
# ============================================================================


class TestZeroShotClassificationProvider:
    """Tests for Zero-Shot classification provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_zero_shot_provider_init(self, model_config):
        """Test Zero-Shot provider initialization."""
        from sentimatrix.providers.models.huggingface import ZeroShotClassificationProvider

        provider = ZeroShotClassificationProvider(config=model_config)

        assert "bart" in provider._model_name.lower()
        assert "mnli" in provider._model_name.lower()

    def test_zero_shot_provider_info(self, model_config):
        """Test Zero-Shot provider info."""
        from sentimatrix.providers.models.huggingface import ZeroShotClassificationProvider

        provider = ZeroShotClassificationProvider(config=model_config)
        info = provider.info

        assert info.name == "zero-shot-classification"
        assert "zero-shot" in info.description.lower()

    def test_zero_shot_default_model(self, model_config):
        """Test Zero-Shot uses correct default model."""
        from sentimatrix.providers.models.huggingface import ZeroShotClassificationProvider

        provider = ZeroShotClassificationProvider(config=model_config)

        assert provider._model_name == ZeroShotClassificationProvider.DEFAULT_MODEL

    def test_zero_shot_registration(self):
        """Test Zero-Shot provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "zero-shot-classification" in providers["model"]


class TestZeroShotResult:
    """Tests for ZeroShotResult dataclass."""

    def test_zero_shot_result_creation(self):
        """Test creating ZeroShotResult."""
        from sentimatrix.providers.models.huggingface import ZeroShotResult

        result = ZeroShotResult(
            text="I love machine learning",
            labels=["technology", "sports", "politics"],
            scores={"technology": 0.85, "sports": 0.10, "politics": 0.05},
            predicted_label="technology",
            confidence=0.85,
            model_name="bart-mnli",
        )

        assert result.text == "I love machine learning"
        assert len(result.labels) == 3
        assert result.predicted_label == "technology"
        assert result.confidence == 0.85

    def test_zero_shot_result_to_dict(self):
        """Test ZeroShotResult serialization."""
        from sentimatrix.providers.models.huggingface import ZeroShotResult

        result = ZeroShotResult(
            text="Test text",
            labels=["a", "b"],
            scores={"a": 0.7, "b": 0.3},
            predicted_label="a",
            confidence=0.7,
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Test text"
        assert result_dict["labels"] == ["a", "b"]
        assert result_dict["predicted_label"] == "a"
        assert result_dict["confidence"] == 0.7


# ============================================================================
# Sentence Embedding Provider Tests
# ============================================================================


class TestSentenceEmbeddingProvider:
    """Tests for Sentence Embedding provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_embedding_provider_init(self, model_config):
        """Test Embedding provider initialization."""
        from sentimatrix.providers.models.huggingface import SentenceEmbeddingProvider

        provider = SentenceEmbeddingProvider(config=model_config)

        assert "minilm" in provider._model_name.lower()

    def test_embedding_provider_mpnet(self, model_config):
        """Test Embedding provider with MPNet model."""
        from sentimatrix.providers.models.huggingface import SentenceEmbeddingProvider

        provider = SentenceEmbeddingProvider(config=model_config, use_mpnet=True)

        assert "mpnet" in provider._model_name.lower()
        assert provider._embedding_dim == 768

    def test_embedding_provider_info(self, model_config):
        """Test Embedding provider info."""
        from sentimatrix.providers.models.huggingface import SentenceEmbeddingProvider

        provider = SentenceEmbeddingProvider(config=model_config)
        info = provider.info

        assert info.name == "sentence-embedding"
        assert "embedding" in info.description.lower()

    def test_embedding_default_dim(self, model_config):
        """Test Embedding default dimension."""
        from sentimatrix.providers.models.huggingface import SentenceEmbeddingProvider

        provider = SentenceEmbeddingProvider(config=model_config)

        assert provider._embedding_dim == 384

    def test_embedding_registration(self):
        """Test Embedding provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "sentence-embedding" in providers["model"]


class TestEmbeddingResult:
    """Tests for EmbeddingResult dataclass."""

    def test_embedding_result_creation(self):
        """Test creating EmbeddingResult."""
        from sentimatrix.providers.models.huggingface import EmbeddingResult

        result = EmbeddingResult(
            text="Hello world",
            embedding=[0.1, 0.2, 0.3] * 128,  # 384 dim
            model_name="minilm",
            dimension=384,
        )

        assert result.text == "Hello world"
        assert len(result.embedding) == 384
        assert result.dimension == 384

    def test_embedding_result_to_dict(self):
        """Test EmbeddingResult serialization."""
        from sentimatrix.providers.models.huggingface import EmbeddingResult

        result = EmbeddingResult(
            text="Test",
            embedding=[0.1, 0.2],
            model_name="test-model",
            dimension=2,
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Test"
        assert result_dict["embedding"] == [0.1, 0.2]
        assert result_dict["dimension"] == 2


# ============================================================================
# Whisper Provider Tests
# ============================================================================


class TestWhisperProvider:
    """Tests for Whisper speech-to-text provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_whisper_provider_init(self, model_config):
        """Test Whisper provider initialization."""
        from sentimatrix.providers.models.huggingface import WhisperProvider

        provider = WhisperProvider(config=model_config)

        assert "whisper" in provider._model_name.lower()
        assert "base" in provider._model_name.lower()

    def test_whisper_provider_sizes(self, model_config):
        """Test Whisper provider with different sizes."""
        from sentimatrix.providers.models.huggingface import WhisperProvider

        for size in ["base", "small", "medium", "large"]:
            provider = WhisperProvider(config=model_config, size=size)
            assert size in provider._model_name.lower() or (size == "large" and "v3" in provider._model_name.lower())

    def test_whisper_provider_info(self, model_config):
        """Test Whisper provider info."""
        from sentimatrix.providers.models.huggingface import WhisperProvider

        provider = WhisperProvider(config=model_config)
        info = provider.info

        assert info.name == "whisper"
        assert "speech" in info.description.lower() or "text" in info.description.lower()

    def test_whisper_models_dict(self):
        """Test Whisper MODELS dictionary."""
        from sentimatrix.providers.models.huggingface import WhisperProvider

        assert len(WhisperProvider.MODELS) == 4
        assert "base" in WhisperProvider.MODELS
        assert "large" in WhisperProvider.MODELS

    def test_whisper_registration(self):
        """Test Whisper provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "whisper" in providers["model"]


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_transcription_result_creation(self):
        """Test creating TranscriptionResult."""
        from sentimatrix.providers.models.huggingface import TranscriptionResult

        result = TranscriptionResult(
            text="Hello, world!",
            audio_path="/path/to/audio.mp3",
            language="en",
            duration_seconds=5.0,
            model_name="whisper-base",
        )

        assert result.text == "Hello, world!"
        assert result.audio_path == "/path/to/audio.mp3"
        assert result.language == "en"
        assert result.duration_seconds == 5.0

    def test_transcription_result_to_dict(self):
        """Test TranscriptionResult serialization."""
        from sentimatrix.providers.models.huggingface import TranscriptionResult

        result = TranscriptionResult(
            text="Test audio",
            audio_path="/test.wav",
            model_name="whisper",
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Test audio"
        assert result_dict["audio_path"] == "/test.wav"


# ============================================================================
# BLIP Captioning Provider Tests
# ============================================================================


class TestBLIPCaptioningProvider:
    """Tests for BLIP image captioning provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_blip_provider_init(self, model_config):
        """Test BLIP provider initialization."""
        from sentimatrix.providers.models.huggingface import BLIPCaptioningProvider

        provider = BLIPCaptioningProvider(config=model_config)

        assert "blip" in provider._model_name.lower()
        assert "captioning" in provider._model_name.lower()

    def test_blip_provider_large(self, model_config):
        """Test BLIP provider with large model."""
        from sentimatrix.providers.models.huggingface import BLIPCaptioningProvider

        provider = BLIPCaptioningProvider(config=model_config, use_large=True)

        assert "large" in provider._model_name.lower()

    def test_blip_provider_info(self, model_config):
        """Test BLIP provider info."""
        from sentimatrix.providers.models.huggingface import BLIPCaptioningProvider

        provider = BLIPCaptioningProvider(config=model_config)
        info = provider.info

        assert info.name == "blip-captioning"
        assert "caption" in info.description.lower() or "image" in info.description.lower()

    def test_blip_default_model(self, model_config):
        """Test BLIP uses correct default model."""
        from sentimatrix.providers.models.huggingface import BLIPCaptioningProvider

        provider = BLIPCaptioningProvider(config=model_config)

        assert provider._model_name == BLIPCaptioningProvider.DEFAULT_MODEL

    def test_blip_registration(self):
        """Test BLIP provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "blip-captioning" in providers["model"]


class TestImageCaptionResult:
    """Tests for ImageCaptionResult dataclass."""

    def test_image_caption_result_creation(self):
        """Test creating ImageCaptionResult."""
        from sentimatrix.providers.models.huggingface import ImageCaptionResult

        result = ImageCaptionResult(
            image_path="/path/to/image.jpg",
            caption="A cat sitting on a couch",
            conditional_caption="A fluffy cat sitting on a red couch",
            model_name="blip-base",
        )

        assert result.image_path == "/path/to/image.jpg"
        assert result.caption == "A cat sitting on a couch"
        assert result.conditional_caption is not None

    def test_image_caption_result_to_dict(self):
        """Test ImageCaptionResult serialization."""
        from sentimatrix.providers.models.huggingface import ImageCaptionResult

        result = ImageCaptionResult(
            image_path="/test.png",
            caption="Test caption",
            model_name="blip",
        )

        result_dict = result.to_dict()

        assert result_dict["image_path"] == "/test.png"
        assert result_dict["caption"] == "Test caption"


# ============================================================================
# OPUS-MT Translation Provider Tests
# ============================================================================


class TestOpusMTTranslationProvider:
    """Tests for OPUS-MT translation provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_opus_mt_provider_init(self, model_config):
        """Test OPUS-MT provider initialization."""
        from sentimatrix.providers.models.huggingface import OpusMTTranslationProvider

        provider = OpusMTTranslationProvider(config=model_config)

        assert "opus-mt" in provider._model_name.lower()
        assert "en-de" in provider._model_name.lower()

    def test_opus_mt_provider_languages(self, model_config):
        """Test OPUS-MT provider with different language pairs."""
        from sentimatrix.providers.models.huggingface import OpusMTTranslationProvider

        provider_en_fr = OpusMTTranslationProvider(
            config=model_config,
            source_lang="en",
            target_lang="fr",
        )
        assert "en-fr" in provider_en_fr._model_name.lower()
        assert provider_en_fr._source_lang == "en"
        assert provider_en_fr._target_lang == "fr"

    def test_opus_mt_provider_info(self, model_config):
        """Test OPUS-MT provider info."""
        from sentimatrix.providers.models.huggingface import OpusMTTranslationProvider

        provider = OpusMTTranslationProvider(config=model_config)
        info = provider.info

        assert info.name == "opus-mt-translation"
        assert "translation" in info.description.lower()

    def test_opus_mt_models_dict(self):
        """Test OPUS-MT MODELS dictionary."""
        from sentimatrix.providers.models.huggingface import OpusMTTranslationProvider

        assert len(OpusMTTranslationProvider.MODELS) >= 12
        assert "en-de" in OpusMTTranslationProvider.MODELS
        assert "en-fr" in OpusMTTranslationProvider.MODELS
        assert "zh-en" in OpusMTTranslationProvider.MODELS

    def test_opus_mt_registration(self):
        """Test OPUS-MT provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "opus-mt-translation" in providers["model"]


class TestTranslationResult:
    """Tests for TranslationResult dataclass."""

    def test_translation_result_creation(self):
        """Test creating TranslationResult."""
        from sentimatrix.providers.models.huggingface import TranslationResult

        result = TranslationResult(
            source_text="Hello, world!",
            translated_text="Hallo, Welt!",
            source_lang="en",
            target_lang="de",
            model_name="opus-mt-en-de",
        )

        assert result.source_text == "Hello, world!"
        assert result.translated_text == "Hallo, Welt!"
        assert result.source_lang == "en"
        assert result.target_lang == "de"

    def test_translation_result_to_dict(self):
        """Test TranslationResult serialization."""
        from sentimatrix.providers.models.huggingface import TranslationResult

        result = TranslationResult(
            source_text="Hello",
            translated_text="Bonjour",
            source_lang="en",
            target_lang="fr",
        )

        result_dict = result.to_dict()

        assert result_dict["source_text"] == "Hello"
        assert result_dict["translated_text"] == "Bonjour"
        assert result_dict["source_lang"] == "en"
        assert result_dict["target_lang"] == "fr"


# ============================================================================
# Extended Provider Registration Tests
# ============================================================================


class TestNewProviderRegistration:
    """Tests for new provider registration."""

    def test_all_new_providers_registered(self):
        """Test that all new providers are registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        # Multilingual
        assert "xlm-roberta-sentiment" in providers["model"]

        # Domain-specific
        assert "finbert" in providers["model"]
        assert "finbert-tone" in providers["model"]

        # Zero-shot
        assert "zero-shot-classification" in providers["model"]

        # Embeddings
        assert "sentence-embedding" in providers["model"]

        # Speech-to-text
        assert "whisper" in providers["model"]

        # Vision
        assert "blip-captioning" in providers["model"]

        # Translation
        assert "opus-mt-translation" in providers["model"]

    def test_total_provider_count(self):
        """Test total number of registered model providers."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        # Should have at least 25 providers (original 11 + 8 + 6 new)
        assert len(providers["model"]) >= 25


# ============================================================================
# Legal-BERT Provider Tests
# ============================================================================


class TestLegalBERTProvider:
    """Tests for Legal-BERT domain provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_legal_bert_provider_init(self, model_config):
        """Test Legal-BERT provider initialization."""
        from sentimatrix.providers.models.huggingface import LegalBERTProvider

        provider = LegalBERTProvider(config=model_config)

        assert "legal-bert" in provider._model_name.lower()

    def test_legal_bert_provider_info(self, model_config):
        """Test Legal-BERT provider info."""
        from sentimatrix.providers.models.huggingface import LegalBERTProvider

        provider = LegalBERTProvider(config=model_config)
        info = provider.info

        assert info.name == "legal-bert"
        assert "legal" in info.description.lower()

    def test_legal_bert_default_model(self, model_config):
        """Test Legal-BERT uses correct default model."""
        from sentimatrix.providers.models.huggingface import LegalBERTProvider

        provider = LegalBERTProvider(config=model_config)

        assert provider._model_name == LegalBERTProvider.DEFAULT_MODEL
        assert "nlpaueb" in provider._model_name

    def test_legal_bert_registration(self):
        """Test Legal-BERT provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "legal-bert" in providers["model"]


# ============================================================================
# SciBERT Provider Tests
# ============================================================================


class TestSciBERTProvider:
    """Tests for SciBERT scientific text provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_scibert_provider_init(self, model_config):
        """Test SciBERT provider initialization."""
        from sentimatrix.providers.models.huggingface import SciBERTProvider

        provider = SciBERTProvider(config=model_config)

        assert "scibert" in provider._model_name.lower()

    def test_scibert_provider_info(self, model_config):
        """Test SciBERT provider info."""
        from sentimatrix.providers.models.huggingface import SciBERTProvider

        provider = SciBERTProvider(config=model_config)
        info = provider.info

        assert info.name == "scibert"
        assert "scientific" in info.description.lower()

    def test_scibert_default_model(self, model_config):
        """Test SciBERT uses correct default model."""
        from sentimatrix.providers.models.huggingface import SciBERTProvider

        provider = SciBERTProvider(config=model_config)

        assert provider._model_name == SciBERTProvider.DEFAULT_MODEL
        assert "allenai" in provider._model_name

    def test_scibert_registration(self):
        """Test SciBERT provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "scibert" in providers["model"]


# ============================================================================
# DeBERTa NLI Provider Tests
# ============================================================================


class TestDeBERTaNLIProvider:
    """Tests for DeBERTa NLI zero-shot provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_deberta_nli_provider_init(self, model_config):
        """Test DeBERTa NLI provider initialization."""
        from sentimatrix.providers.models.huggingface import DeBERTaNLIProvider

        provider = DeBERTaNLIProvider(config=model_config)

        assert "deberta" in provider._model_name.lower()
        assert "mnli" in provider._model_name.lower()

    def test_deberta_nli_provider_info(self, model_config):
        """Test DeBERTa NLI provider info."""
        from sentimatrix.providers.models.huggingface import DeBERTaNLIProvider

        provider = DeBERTaNLIProvider(config=model_config)
        info = provider.info

        assert info.name == "deberta-nli"
        assert "zero-shot" in info.description.lower()

    def test_deberta_nli_default_model(self, model_config):
        """Test DeBERTa NLI uses correct default model."""
        from sentimatrix.providers.models.huggingface import DeBERTaNLIProvider

        provider = DeBERTaNLIProvider(config=model_config)

        assert provider._model_name == DeBERTaNLIProvider.DEFAULT_MODEL
        assert "MoritzLaurer" in provider._model_name

    def test_deberta_nli_registration(self):
        """Test DeBERTa NLI provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "deberta-nli" in providers["model"]


# ============================================================================
# BGE Embedding Provider Tests
# ============================================================================


class TestBGEEmbeddingProvider:
    """Tests for BGE embedding provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_bge_provider_init(self, model_config):
        """Test BGE provider initialization."""
        from sentimatrix.providers.models.huggingface import BGEEmbeddingProvider

        provider = BGEEmbeddingProvider(config=model_config)

        assert "bge" in provider._model_name.lower()
        assert "small" in provider._model_name.lower()

    def test_bge_provider_large(self, model_config):
        """Test BGE provider with large model."""
        from sentimatrix.providers.models.huggingface import BGEEmbeddingProvider

        provider = BGEEmbeddingProvider(config=model_config, use_large=True)

        assert "large" in provider._model_name.lower()
        assert provider._embedding_dim == 1024

    def test_bge_provider_info(self, model_config):
        """Test BGE provider info."""
        from sentimatrix.providers.models.huggingface import BGEEmbeddingProvider

        provider = BGEEmbeddingProvider(config=model_config)
        info = provider.info

        assert info.name == "bge-embedding"
        assert "retrieval" in info.description.lower() or "embedding" in info.description.lower()

    def test_bge_default_dim(self, model_config):
        """Test BGE default dimension."""
        from sentimatrix.providers.models.huggingface import BGEEmbeddingProvider

        provider = BGEEmbeddingProvider(config=model_config)

        assert provider._embedding_dim == 384

    def test_bge_registration(self):
        """Test BGE provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "bge-embedding" in providers["model"]


# ============================================================================
# E5 Embedding Provider Tests
# ============================================================================


class TestE5EmbeddingProvider:
    """Tests for E5 embedding provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_e5_provider_init(self, model_config):
        """Test E5 provider initialization."""
        from sentimatrix.providers.models.huggingface import E5EmbeddingProvider

        provider = E5EmbeddingProvider(config=model_config)

        assert "e5" in provider._model_name.lower()

    def test_e5_provider_large(self, model_config):
        """Test E5 provider with large model."""
        from sentimatrix.providers.models.huggingface import E5EmbeddingProvider

        provider = E5EmbeddingProvider(config=model_config, use_large=True)

        assert "large" in provider._model_name.lower()
        assert provider._embedding_dim == 1024

    def test_e5_provider_info(self, model_config):
        """Test E5 provider info."""
        from sentimatrix.providers.models.huggingface import E5EmbeddingProvider

        provider = E5EmbeddingProvider(config=model_config)
        info = provider.info

        assert info.name == "e5-embedding"
        assert "retrieval" in info.description.lower() or "embedding" in info.description.lower()

    def test_e5_default_dim(self, model_config):
        """Test E5 default dimension."""
        from sentimatrix.providers.models.huggingface import E5EmbeddingProvider

        provider = E5EmbeddingProvider(config=model_config)

        assert provider._embedding_dim == 768

    def test_e5_registration(self):
        """Test E5 provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "e5-embedding" in providers["model"]


# ============================================================================
# mBART Translation Provider Tests
# ============================================================================


class TestMBartTranslationProvider:
    """Tests for mBART-50 translation provider."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(device="cpu", cache_models=False)

    def test_mbart_provider_init(self, model_config):
        """Test mBART provider initialization."""
        from sentimatrix.providers.models.huggingface import MBartTranslationProvider

        provider = MBartTranslationProvider(config=model_config)

        assert "mbart" in provider._model_name.lower()
        assert "50" in provider._model_name

    def test_mbart_provider_languages(self, model_config):
        """Test mBART provider with different language pairs."""
        from sentimatrix.providers.models.huggingface import MBartTranslationProvider

        provider = MBartTranslationProvider(
            config=model_config,
            source_lang="en",
            target_lang="de",
        )
        assert provider._source_lang == "en"
        assert provider._target_lang == "de"
        assert provider._source_lang_code == "en_XX"
        assert provider._target_lang_code == "de_DE"

    def test_mbart_provider_info(self, model_config):
        """Test mBART provider info."""
        from sentimatrix.providers.models.huggingface import MBartTranslationProvider

        provider = MBartTranslationProvider(config=model_config)
        info = provider.info

        assert info.name == "mbart-translation"
        assert "translation" in info.description.lower()

    def test_mbart_language_codes(self):
        """Test mBART language code mapping."""
        from sentimatrix.providers.models.huggingface import MBartTranslationProvider

        assert len(MBartTranslationProvider.LANGUAGE_CODES) >= 50
        assert "en" in MBartTranslationProvider.LANGUAGE_CODES
        assert "zh" in MBartTranslationProvider.LANGUAGE_CODES
        assert "ja" in MBartTranslationProvider.LANGUAGE_CODES

    def test_mbart_registration(self):
        """Test mBART provider is registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        assert "mbart-translation" in providers["model"]


class TestMBartTranslationResult:
    """Tests for MBartTranslationResult dataclass."""

    def test_mbart_result_creation(self):
        """Test creating MBartTranslationResult."""
        from sentimatrix.providers.models.huggingface import MBartTranslationResult

        result = MBartTranslationResult(
            source_text="Hello, world!",
            translated_text="Hallo, Welt!",
            source_lang="en",
            target_lang="de",
            model_name="mbart-50",
        )

        assert result.source_text == "Hello, world!"
        assert result.translated_text == "Hallo, Welt!"
        assert result.source_lang == "en"
        assert result.target_lang == "de"

    def test_mbart_result_to_dict(self):
        """Test MBartTranslationResult serialization."""
        from sentimatrix.providers.models.huggingface import MBartTranslationResult

        result = MBartTranslationResult(
            source_text="Hello",
            translated_text="Bonjour",
            source_lang="en",
            target_lang="fr",
        )

        result_dict = result.to_dict()

        assert result_dict["source_text"] == "Hello"
        assert result_dict["translated_text"] == "Bonjour"


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestNewHelperFunctions:
    """Tests for new helper functions."""

    def test_get_available_domain_models(self):
        """Test getting available domain-specific models."""
        from sentimatrix.providers.models.huggingface import get_available_domain_models

        models = get_available_domain_models()

        assert isinstance(models, dict)
        assert len(models) >= 4
        assert "legal-bert" in models
        assert "scibert" in models
        assert "finbert" in models

    def test_get_available_zero_shot_models(self):
        """Test getting available zero-shot models."""
        from sentimatrix.providers.models.huggingface import get_available_zero_shot_models

        models = get_available_zero_shot_models()

        assert isinstance(models, dict)
        assert len(models) >= 2
        assert "bart-mnli" in models
        assert "deberta-nli" in models

    def test_get_available_embedding_models_extended(self):
        """Test getting available embedding models includes new ones."""
        from sentimatrix.providers.models.huggingface import get_available_embedding_models

        models = get_available_embedding_models()

        assert isinstance(models, dict)
        assert len(models) >= 6
        assert "bge-small" in models
        assert "bge-large" in models
        assert "e5-base" in models
        assert "e5-large" in models


# ============================================================================
# Complete Provider Registration Tests
# ============================================================================


class TestCompleteProviderRegistration:
    """Tests for complete provider registration."""

    def test_all_stage16_providers_registered(self):
        """Test that all Stage 16 providers are registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        # Domain-specific
        assert "legal-bert" in providers["model"]
        assert "scibert" in providers["model"]

        # Zero-shot
        assert "deberta-nli" in providers["model"]

        # Embeddings
        assert "bge-embedding" in providers["model"]
        assert "e5-embedding" in providers["model"]

        # Translation
        assert "mbart-translation" in providers["model"]

    def test_final_provider_count(self):
        """Test final number of registered model providers."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        # Should have at least 29 providers (including 4 new ones)
        assert len(providers["model"]) >= 29

    def test_all_final_providers_registered(self):
        """Test that all final Stage 16 providers are registered."""
        from sentimatrix.providers.base import ProviderRegistry

        registry = ProviderRegistry()
        providers = registry.list_providers("model")

        # Final 4 providers
        assert "cross-encoder-nli" in providers["model"]
        assert "blip2" in providers["model"]
        assert "deep-translator" in providers["model"]
        assert "llava" in providers["model"]


# ============================================================================
# CROSS-ENCODER NLI PROVIDER TESTS
# ============================================================================


class TestCrossEncoderNLIResult:
    """Tests for CrossEncoderNLIResult dataclass."""

    def test_result_creation(self):
        """Test creating CrossEncoderNLIResult."""
        from sentimatrix.providers.models.huggingface import CrossEncoderNLIResult

        result = CrossEncoderNLIResult(
            sentence1="A man is eating pizza.",
            sentence2="A man is eating food.",
            label="entailment",
            scores={"contradiction": 0.01, "entailment": 0.95, "neutral": 0.04},
            confidence=0.95,
            model_name="cross-encoder/nli-deberta-v3-base",
        )

        assert result.sentence1 == "A man is eating pizza."
        assert result.sentence2 == "A man is eating food."
        assert result.label == "entailment"
        assert result.confidence == 0.95

    def test_result_to_dict(self):
        """Test converting CrossEncoderNLIResult to dict."""
        from sentimatrix.providers.models.huggingface import CrossEncoderNLIResult

        result = CrossEncoderNLIResult(
            sentence1="The dog is sleeping.",
            sentence2="The cat is running.",
            label="neutral",
            scores={"contradiction": 0.1, "entailment": 0.1, "neutral": 0.8},
            confidence=0.8,
            model_name="test-model",
        )

        result_dict = result.to_dict()

        assert result_dict["sentence1"] == "The dog is sleeping."
        assert result_dict["sentence2"] == "The cat is running."
        assert result_dict["label"] == "neutral"
        assert result_dict["confidence"] == 0.8


class TestCrossEncoderNLIProvider:
    """Tests for CrossEncoderNLIProvider."""

    def test_provider_creation(self):
        """Test creating CrossEncoderNLI provider."""
        from sentimatrix.providers.models.huggingface import CrossEncoderNLIProvider

        provider = CrossEncoderNLIProvider()

        assert provider._model_name == "cross-encoder/nli-deberta-v3-base"
        assert not provider._initialized

    def test_provider_custom_model(self):
        """Test creating provider with custom model."""
        from sentimatrix.providers.models.huggingface import (
            CrossEncoderNLIProvider,
            ModelConfig,
        )

        config = ModelConfig()
        provider = CrossEncoderNLIProvider(
            config=config,
            model_name="cross-encoder/nli-deberta-v3-small",
        )

        assert provider._model_name == "cross-encoder/nli-deberta-v3-small"

    def test_provider_info(self):
        """Test CrossEncoderNLI provider info."""
        from sentimatrix.providers.models.huggingface import CrossEncoderNLIProvider

        provider = CrossEncoderNLIProvider()
        info = provider.info

        assert info.name == "cross-encoder-nli"
        assert info.version == "1.0.0"
        assert "cross-encoder" in info.description.lower() or "nli" in info.description.lower()

    def test_labels_defined(self):
        """Test NLI labels are defined."""
        from sentimatrix.providers.models.huggingface import CrossEncoderNLIProvider

        provider = CrossEncoderNLIProvider()

        assert "contradiction" in provider.LABELS
        assert "entailment" in provider.LABELS
        assert "neutral" in provider.LABELS


# ============================================================================
# BLIP2 PROVIDER TESTS
# ============================================================================


class TestBLIP2Result:
    """Tests for BLIP2Result dataclass."""

    def test_caption_result_creation(self):
        """Test creating BLIP2Result for captioning."""
        from sentimatrix.providers.models.huggingface import BLIP2Result

        result = BLIP2Result(
            image_path="/path/to/image.jpg",
            caption="A beautiful sunset over the ocean.",
            model_name="Salesforce/blip2-opt-2.7b",
        )

        assert result.image_path == "/path/to/image.jpg"
        assert result.caption == "A beautiful sunset over the ocean."
        assert result.answer is None

    def test_vqa_result_creation(self):
        """Test creating BLIP2Result for VQA."""
        from sentimatrix.providers.models.huggingface import BLIP2Result

        result = BLIP2Result(
            image_path="/path/to/image.jpg",
            question="What color is the sky?",
            answer="Blue",
            model_name="Salesforce/blip2-opt-2.7b",
        )

        assert result.image_path == "/path/to/image.jpg"
        assert result.question == "What color is the sky?"
        assert result.answer == "Blue"

    def test_result_to_dict(self):
        """Test converting BLIP2Result to dict."""
        from sentimatrix.providers.models.huggingface import BLIP2Result

        result = BLIP2Result(
            image_path="/test/image.png",
            caption="Test caption",
            model_name="test-model",
        )

        result_dict = result.to_dict()

        assert result_dict["image_path"] == "/test/image.png"
        assert result_dict["caption"] == "Test caption"


class TestBLIP2Provider:
    """Tests for BLIP2Provider."""

    def test_provider_creation(self):
        """Test creating BLIP2 provider."""
        from sentimatrix.providers.models.huggingface import BLIP2Provider

        provider = BLIP2Provider()

        assert provider._model_name == "Salesforce/blip2-opt-2.7b"
        assert not provider._initialized

    def test_provider_with_quantization(self):
        """Test creating provider with 8-bit quantization."""
        from sentimatrix.providers.models.huggingface import BLIP2Provider, ModelConfig

        config = ModelConfig(use_quantization=True)
        provider = BLIP2Provider(config=config, load_in_8bit=True)

        assert provider._load_in_8bit is True

    def test_provider_info(self):
        """Test BLIP2 provider info."""
        from sentimatrix.providers.models.huggingface import BLIP2Provider

        provider = BLIP2Provider()
        info = provider.info

        assert info.name == "blip2"
        assert info.version == "1.0.0"
        assert info.capabilities.gpu_support is True

    def test_provider_custom_model(self):
        """Test creating provider with custom model."""
        from sentimatrix.providers.models.huggingface import BLIP2Provider

        provider = BLIP2Provider(model_name="Salesforce/blip2-flan-t5-xl")

        assert provider._model_name == "Salesforce/blip2-flan-t5-xl"


# ============================================================================
# DEEP TRANSLATOR PROVIDER TESTS
# ============================================================================


class TestDeepTranslatorResult:
    """Tests for DeepTranslatorResult dataclass."""

    def test_result_creation(self):
        """Test creating DeepTranslatorResult."""
        from sentimatrix.providers.models.huggingface import DeepTranslatorResult

        result = DeepTranslatorResult(
            source_text="Hello world",
            translated_text="Bonjour le monde",
            source_lang="en",
            target_lang="fr",
            translator="google",
        )

        assert result.source_text == "Hello world"
        assert result.translated_text == "Bonjour le monde"
        assert result.source_lang == "en"
        assert result.target_lang == "fr"
        assert result.translator == "google"

    def test_result_to_dict(self):
        """Test converting DeepTranslatorResult to dict."""
        from sentimatrix.providers.models.huggingface import DeepTranslatorResult

        result = DeepTranslatorResult(
            source_text="Test",
            translated_text="Teste",
            source_lang="en",
            target_lang="pt",
            translator="google",
        )

        result_dict = result.to_dict()

        assert result_dict["source_text"] == "Test"
        assert result_dict["translated_text"] == "Teste"
        assert result_dict["translator"] == "google"


class TestDeepTranslatorProvider:
    """Tests for DeepTranslatorProvider."""

    def test_provider_creation(self):
        """Test creating DeepTranslator provider."""
        from sentimatrix.providers.models.huggingface import DeepTranslatorProvider

        provider = DeepTranslatorProvider()

        assert provider._source_lang == "auto"
        assert provider._target_lang == "en"
        assert provider._translator_name == "google"
        assert not provider._initialized

    def test_provider_custom_languages(self):
        """Test creating provider with custom languages."""
        from sentimatrix.providers.models.huggingface import DeepTranslatorProvider

        provider = DeepTranslatorProvider(
            source_lang="en",
            target_lang="es",
        )

        assert provider._source_lang == "en"
        assert provider._target_lang == "es"

    def test_provider_info(self):
        """Test DeepTranslator provider info."""
        from sentimatrix.providers.models.huggingface import DeepTranslatorProvider

        provider = DeepTranslatorProvider()
        info = provider.info

        assert info.name == "deep-translator"
        assert info.version == "1.0.0"
        assert info.capabilities.batch_processing is True
        assert info.capabilities.gpu_support is False

    def test_supported_translators(self):
        """Test supported translators list."""
        from sentimatrix.providers.models.huggingface import DeepTranslatorProvider

        assert "google" in DeepTranslatorProvider.SUPPORTED_TRANSLATORS
        assert "mymemory" in DeepTranslatorProvider.SUPPORTED_TRANSLATORS

    def test_is_initialized_property(self):
        """Test is_initialized property."""
        from sentimatrix.providers.models.huggingface import DeepTranslatorProvider

        provider = DeepTranslatorProvider()

        assert provider.is_initialized is False


# ============================================================================
# LLAVA PROVIDER TESTS
# ============================================================================


class TestLLaVAResult:
    """Tests for LLaVAResult dataclass."""

    def test_result_creation(self):
        """Test creating LLaVAResult."""
        from sentimatrix.providers.models.huggingface import LLaVAResult

        result = LLaVAResult(
            image_path="/path/to/image.jpg",
            response="This image shows a cat sitting on a couch.",
            prompt="Describe this image in detail.",
            model_name="llava",
        )

        assert result.image_path == "/path/to/image.jpg"
        assert "cat" in result.response
        assert result.prompt == "Describe this image in detail."
        assert result.model_name == "llava"

    def test_result_to_dict(self):
        """Test converting LLaVAResult to dict."""
        from sentimatrix.providers.models.huggingface import LLaVAResult

        result = LLaVAResult(
            image_path="/test/image.png",
            response="A dog running in a field.",
            prompt="What is in this image?",
            model_name="llava:13b",
        )

        result_dict = result.to_dict()

        assert result_dict["image_path"] == "/test/image.png"
        assert result_dict["response"] == "A dog running in a field."
        assert result_dict["model_name"] == "llava:13b"


class TestLLaVAProvider:
    """Tests for LLaVAProvider."""

    def test_provider_creation(self):
        """Test creating LLaVA provider."""
        from sentimatrix.providers.models.huggingface import LLaVAProvider

        provider = LLaVAProvider()

        assert provider._model_name == "llava"
        assert provider._base_url == "http://localhost:11434"
        assert not provider._initialized

    def test_provider_custom_model(self):
        """Test creating provider with custom model."""
        from sentimatrix.providers.models.huggingface import LLaVAProvider

        provider = LLaVAProvider(model_name="llava:13b")

        assert provider._model_name == "llava:13b"

    def test_provider_custom_url(self):
        """Test creating provider with custom URL."""
        from sentimatrix.providers.models.huggingface import LLaVAProvider

        provider = LLaVAProvider(base_url="http://192.168.1.100:11434")

        assert provider._base_url == "http://192.168.1.100:11434"

    def test_provider_info(self):
        """Test LLaVA provider info."""
        from sentimatrix.providers.models.huggingface import LLaVAProvider

        provider = LLaVAProvider()
        info = provider.info

        assert info.name == "llava"
        assert info.version == "1.0.0"
        assert info.capabilities.gpu_support is True
        assert "llava" in info.supported_models
        assert "llava:13b" in info.supported_models

    def test_is_initialized_property(self):
        """Test is_initialized property."""
        from sentimatrix.providers.models.huggingface import LLaVAProvider

        provider = LLaVAProvider()

        assert provider.is_initialized is False

    def test_default_model_constant(self):
        """Test DEFAULT_MODEL constant."""
        from sentimatrix.providers.models.huggingface import LLaVAProvider

        assert LLaVAProvider.DEFAULT_MODEL == "llava"
