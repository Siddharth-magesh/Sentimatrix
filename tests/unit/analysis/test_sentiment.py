"""
Unit Tests for Sentiment Analysis Module

Tests the sentiment analysis functionality including:
- SentimentResult dataclass
- BatchSentimentResult dataclass
- SentimentAnalyzer class
- Label normalization
- Input validation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentimatrix.core.config import ModelConfig, SentimatrixConfig
from sentimatrix.core.exceptions import InvalidInputError, SentimatrixError


class TestSentimentLabel:
    """Tests for SentimentLabel enum."""

    def test_sentiment_labels_exist(self):
        """Test that all sentiment labels are defined."""
        from sentimatrix.analysis.sentiment import SentimentLabel

        assert SentimentLabel.POSITIVE
        assert SentimentLabel.NEGATIVE
        assert SentimentLabel.NEUTRAL
        assert SentimentLabel.VERY_POSITIVE
        assert SentimentLabel.VERY_NEGATIVE

    def test_sentiment_label_values(self):
        """Test sentiment label string values."""
        from sentimatrix.analysis.sentiment import SentimentLabel

        assert SentimentLabel.POSITIVE.value == "positive"
        assert SentimentLabel.NEGATIVE.value == "negative"
        assert SentimentLabel.NEUTRAL.value == "neutral"


class TestSentimentClass:
    """Tests for SentimentClass enum."""

    def test_classification_modes(self):
        """Test classification mode values."""
        from sentimatrix.analysis.sentiment import SentimentClass

        assert SentimentClass.THREE_CLASS.value == "three_class"
        assert SentimentClass.FIVE_CLASS.value == "five_class"
        assert SentimentClass.BINARY.value == "binary"


class TestSentimentResult:
    """Tests for SentimentResult dataclass."""

    def test_create_sentiment_result(self):
        """Test creating a sentiment result."""
        from sentimatrix.analysis.sentiment import SentimentLabel, SentimentResult

        result = SentimentResult(
            text="Great product!",
            sentiment=SentimentLabel.POSITIVE,
            confidence=0.95,
            raw_label="LABEL_2",
            raw_score=0.95,
            all_scores={"negative": 0.02, "neutral": 0.03, "positive": 0.95},
            model_name="test-model",
            processing_time_ms=10.5,
        )

        assert result.text == "Great product!"
        assert result.sentiment == SentimentLabel.POSITIVE
        assert result.confidence == 0.95
        assert result.processing_time_ms == 10.5

    def test_is_positive(self):
        """Test is_positive property."""
        from sentimatrix.analysis.sentiment import SentimentLabel, SentimentResult

        positive_result = SentimentResult(
            text="Great!",
            sentiment=SentimentLabel.POSITIVE,
            confidence=0.9,
        )
        assert positive_result.is_positive is True
        assert positive_result.is_negative is False
        assert positive_result.is_neutral is False

        very_positive_result = SentimentResult(
            text="Amazing!",
            sentiment=SentimentLabel.VERY_POSITIVE,
            confidence=0.9,
        )
        assert very_positive_result.is_positive is True

    def test_is_negative(self):
        """Test is_negative property."""
        from sentimatrix.analysis.sentiment import SentimentLabel, SentimentResult

        negative_result = SentimentResult(
            text="Terrible!",
            sentiment=SentimentLabel.NEGATIVE,
            confidence=0.9,
        )
        assert negative_result.is_negative is True
        assert negative_result.is_positive is False

        very_negative_result = SentimentResult(
            text="Worst ever!",
            sentiment=SentimentLabel.VERY_NEGATIVE,
            confidence=0.9,
        )
        assert very_negative_result.is_negative is True

    def test_is_neutral(self):
        """Test is_neutral property."""
        from sentimatrix.analysis.sentiment import SentimentLabel, SentimentResult

        neutral_result = SentimentResult(
            text="It's okay.",
            sentiment=SentimentLabel.NEUTRAL,
            confidence=0.8,
        )
        assert neutral_result.is_neutral is True
        assert neutral_result.is_positive is False
        assert neutral_result.is_negative is False

    def test_polarity(self):
        """Test polarity calculation."""
        from sentimatrix.analysis.sentiment import SentimentLabel, SentimentResult

        positive = SentimentResult(
            text="Great!",
            sentiment=SentimentLabel.POSITIVE,
            confidence=1.0,
        )
        assert positive.polarity == 0.5

        negative = SentimentResult(
            text="Bad!",
            sentiment=SentimentLabel.NEGATIVE,
            confidence=1.0,
        )
        assert negative.polarity == -0.5

        neutral = SentimentResult(
            text="Okay",
            sentiment=SentimentLabel.NEUTRAL,
            confidence=1.0,
        )
        assert neutral.polarity == 0.0

        very_positive = SentimentResult(
            text="Amazing!",
            sentiment=SentimentLabel.VERY_POSITIVE,
            confidence=1.0,
        )
        assert very_positive.polarity == 1.0

    def test_to_dict(self):
        """Test result serialization."""
        from sentimatrix.analysis.sentiment import SentimentLabel, SentimentResult

        result = SentimentResult(
            text="Great product!",
            sentiment=SentimentLabel.POSITIVE,
            confidence=0.95,
            raw_label="positive",
            raw_score=0.95,
            all_scores={"negative": 0.02, "neutral": 0.03, "positive": 0.95},
            model_name="test-model",
            processing_time_ms=10.5,
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Great product!"
        assert result_dict["sentiment"] == "positive"
        assert result_dict["confidence"] == 0.95
        assert result_dict["is_positive"] is True
        assert result_dict["is_negative"] is False
        assert result_dict["polarity"] == pytest.approx(0.475, rel=0.01)


class TestBatchSentimentResult:
    """Tests for BatchSentimentResult dataclass."""

    def test_empty_batch_result(self):
        """Test empty batch result."""
        from sentimatrix.analysis.sentiment import BatchSentimentResult

        batch = BatchSentimentResult(results=[])

        assert batch.total_count == 0
        assert batch.positive_count == 0
        assert batch.negative_count == 0
        assert batch.neutral_count == 0

    def test_batch_statistics(self):
        """Test batch statistics calculation."""
        from sentimatrix.analysis.sentiment import (
            BatchSentimentResult,
            SentimentLabel,
            SentimentResult,
        )

        results = [
            SentimentResult(text="Great!", sentiment=SentimentLabel.POSITIVE, confidence=0.9),
            SentimentResult(text="Bad!", sentiment=SentimentLabel.NEGATIVE, confidence=0.8),
            SentimentResult(text="Okay", sentiment=SentimentLabel.NEUTRAL, confidence=0.7),
            SentimentResult(text="Amazing!", sentiment=SentimentLabel.VERY_POSITIVE, confidence=0.95),
        ]

        batch = BatchSentimentResult(results=results)

        assert batch.total_count == 4
        assert batch.positive_count == 2  # POSITIVE + VERY_POSITIVE
        assert batch.negative_count == 1
        assert batch.neutral_count == 1
        assert batch.average_confidence == pytest.approx(0.8375, rel=0.01)

    def test_batch_ratios(self):
        """Test batch ratio calculations."""
        from sentimatrix.analysis.sentiment import (
            BatchSentimentResult,
            SentimentLabel,
            SentimentResult,
        )

        results = [
            SentimentResult(text="Great!", sentiment=SentimentLabel.POSITIVE, confidence=0.9),
            SentimentResult(text="Bad!", sentiment=SentimentLabel.NEGATIVE, confidence=0.8),
        ]

        batch = BatchSentimentResult(results=results)

        assert batch.positive_ratio == 0.5
        assert batch.negative_ratio == 0.5
        assert batch.neutral_ratio == 0.0

    def test_average_polarity(self):
        """Test average polarity calculation."""
        from sentimatrix.analysis.sentiment import (
            BatchSentimentResult,
            SentimentLabel,
            SentimentResult,
        )

        results = [
            SentimentResult(text="Great!", sentiment=SentimentLabel.POSITIVE, confidence=1.0),
            SentimentResult(text="Bad!", sentiment=SentimentLabel.NEGATIVE, confidence=1.0),
        ]

        batch = BatchSentimentResult(results=results)

        assert batch.average_polarity == 0.0  # 0.5 + (-0.5) / 2

    def test_get_summary(self):
        """Test getting summary without results."""
        from sentimatrix.analysis.sentiment import (
            BatchSentimentResult,
            SentimentLabel,
            SentimentResult,
        )

        results = [
            SentimentResult(text="Great!", sentiment=SentimentLabel.POSITIVE, confidence=0.9),
        ]

        batch = BatchSentimentResult(results=results)
        summary = batch.get_summary()

        assert "results" not in summary
        assert summary["total_count"] == 1
        assert summary["positive_count"] == 1


class TestLabelMapping:
    """Tests for label mapping and normalization."""

    def test_label_mapping_exists(self):
        """Test that label mapping is defined."""
        from sentimatrix.analysis.sentiment import LABEL_MAPPING

        assert "LABEL_0" in LABEL_MAPPING
        assert "LABEL_1" in LABEL_MAPPING
        assert "LABEL_2" in LABEL_MAPPING
        assert "positive" in LABEL_MAPPING
        assert "negative" in LABEL_MAPPING
        assert "neutral" in LABEL_MAPPING

    def test_five_star_mapping(self):
        """Test five-star rating mapping."""
        from sentimatrix.analysis.sentiment import LABEL_MAPPING, SentimentLabel

        assert LABEL_MAPPING["1 star"] == SentimentLabel.VERY_NEGATIVE
        assert LABEL_MAPPING["2 stars"] == SentimentLabel.NEGATIVE
        assert LABEL_MAPPING["3 stars"] == SentimentLabel.NEUTRAL
        assert LABEL_MAPPING["4 stars"] == SentimentLabel.POSITIVE
        assert LABEL_MAPPING["5 stars"] == SentimentLabel.VERY_POSITIVE


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device="cpu",
            cache_models=False,
        )

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        from sentimatrix.providers.base import PredictionResult

        provider = AsyncMock()
        provider.initialize = AsyncMock()
        provider.close = AsyncMock()
        provider.predict = AsyncMock(
            return_value=PredictionResult(
                label="positive",
                score=0.95,
                confidence=0.95,
                all_scores={"negative": 0.02, "neutral": 0.03, "positive": 0.95},
                model_name="test-model",
                processing_time_ms=10.0,
            )
        )
        provider.predict_batch = AsyncMock(
            return_value=[
                PredictionResult(
                    label="positive",
                    score=0.95,
                    confidence=0.95,
                    all_scores={},
                    model_name="test-model",
                    processing_time_ms=5.0,
                ),
                PredictionResult(
                    label="negative",
                    score=0.90,
                    confidence=0.90,
                    all_scores={},
                    model_name="test-model",
                    processing_time_ms=5.0,
                ),
            ]
        )
        return provider

    def test_analyzer_init_with_model_config(self, model_config):
        """Test analyzer initialization with ModelConfig."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        assert analyzer._model_name == model_config.sentiment_model
        assert not analyzer.is_initialized

    def test_analyzer_init_with_sentimatrix_config(self):
        """Test analyzer initialization with SentimatrixConfig."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        config = SentimatrixConfig()
        analyzer = SentimentAnalyzer(config=config)

        assert analyzer._model_name == config.models.sentiment_model

    def test_analyzer_init_with_custom_model(self, model_config):
        """Test analyzer initialization with custom model."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        custom_model = "custom/model-name"
        analyzer = SentimentAnalyzer(config=model_config, model_name=custom_model)

        assert analyzer._model_name == custom_model

    def test_analyzer_classification_mode(self, model_config):
        """Test analyzer classification mode."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer, SentimentClass

        analyzer = SentimentAnalyzer(
            config=model_config,
            classification_mode=SentimentClass.FIVE_CLASS,
        )

        assert analyzer.classification_mode == SentimentClass.FIVE_CLASS

    def test_normalize_label_direct(self, model_config):
        """Test direct label normalization."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer, SentimentLabel

        analyzer = SentimentAnalyzer(config=model_config)

        assert analyzer._normalize_label("positive") == SentimentLabel.POSITIVE
        assert analyzer._normalize_label("negative") == SentimentLabel.NEGATIVE
        assert analyzer._normalize_label("neutral") == SentimentLabel.NEUTRAL

    def test_normalize_label_model_output(self, model_config):
        """Test model output label normalization."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer, SentimentLabel

        analyzer = SentimentAnalyzer(config=model_config)

        assert analyzer._normalize_label("LABEL_0") == SentimentLabel.NEGATIVE
        assert analyzer._normalize_label("LABEL_1") == SentimentLabel.NEUTRAL
        assert analyzer._normalize_label("LABEL_2") == SentimentLabel.POSITIVE

    def test_normalize_label_case_insensitive(self, model_config):
        """Test case-insensitive label normalization."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer, SentimentLabel

        analyzer = SentimentAnalyzer(config=model_config)

        assert analyzer._normalize_label("POSITIVE") == SentimentLabel.POSITIVE
        assert analyzer._normalize_label("Negative") == SentimentLabel.NEGATIVE
        assert analyzer._normalize_label("NEUTRAL") == SentimentLabel.NEUTRAL

    def test_normalize_label_inference(self, model_config):
        """Test label inference from name."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer, SentimentLabel

        analyzer = SentimentAnalyzer(config=model_config)

        assert analyzer._normalize_label("very_positive") == SentimentLabel.VERY_POSITIVE
        assert analyzer._normalize_label("strong_negative") == SentimentLabel.VERY_NEGATIVE

    def test_normalize_label_unknown(self, model_config):
        """Test unknown label normalization."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer, SentimentLabel

        analyzer = SentimentAnalyzer(config=model_config)

        assert analyzer._normalize_label("unknown_label") == SentimentLabel.NEUTRAL

    def test_validate_text_valid(self, model_config):
        """Test text validation with valid input."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        result = analyzer._validate_text("  Valid text  ")
        assert result == "Valid text"

    def test_validate_text_none_raises_error(self, model_config):
        """Test text validation with None."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        with pytest.raises(InvalidInputError) as exc_info:
            analyzer._validate_text(None)

        assert "cannot be None" in str(exc_info.value)

    def test_validate_text_empty_raises_error(self, model_config):
        """Test text validation with empty string."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        with pytest.raises(InvalidInputError) as exc_info:
            analyzer._validate_text("   ")

        assert "cannot be empty" in str(exc_info.value)

    def test_validate_text_wrong_type_raises_error(self, model_config):
        """Test text validation with wrong type."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        with pytest.raises(InvalidInputError) as exc_info:
            analyzer._validate_text(123)

        assert "must be a string" in str(exc_info.value)

    def test_validate_text_too_long_raises_error(self, model_config):
        """Test text validation with too long input."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        long_text = "a" * 100001
        with pytest.raises(InvalidInputError) as exc_info:
            analyzer._validate_text(long_text)

        assert "too long" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_not_initialized_raises_error(self, model_config):
        """Test analyze without initialization."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        with pytest.raises(SentimatrixError) as exc_info:
            await analyzer.analyze("Test text")

        assert "not initialized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_analyze_batch_not_initialized_raises_error(self, model_config):
        """Test analyze_batch without initialization."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        with pytest.raises(SentimatrixError):
            await analyzer.analyze_batch(["Test text"])

    @pytest.mark.asyncio
    async def test_analyze_with_mock_provider(self, model_config, mock_provider):
        """Test analyze with mocked provider."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer, SentimentLabel

        analyzer = SentimentAnalyzer(config=model_config)
        analyzer._provider = mock_provider
        analyzer._initialized = True

        result = await analyzer.analyze("Great product!")

        assert result.sentiment == SentimentLabel.POSITIVE
        assert result.confidence == 0.95
        assert result.text == "Great product!"
        mock_provider.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_batch_with_mock_provider(self, model_config, mock_provider):
        """Test analyze_batch with mocked provider."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)
        analyzer._provider = mock_provider
        analyzer._initialized = True

        texts = ["Great product!", "Terrible service!"]
        result = await analyzer.analyze_batch(texts)

        assert result.total_count == 2
        assert result.positive_count == 1
        assert result.negative_count == 1
        mock_provider.predict_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_batch_empty_list(self, model_config, mock_provider):
        """Test analyze_batch with empty list."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)
        analyzer._provider = mock_provider
        analyzer._initialized = True

        result = await analyzer.analyze_batch([])

        assert result.total_count == 0
        assert result.results == []

    @pytest.mark.asyncio
    async def test_get_quick_sentiment(self, model_config, mock_provider):
        """Test get_quick_sentiment convenience method."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)
        analyzer._provider = mock_provider
        analyzer._initialized = True

        label, score = await analyzer.get_quick_sentiment("Great product!")

        assert label == "positive"
        assert score == 0.95

    @pytest.mark.asyncio
    async def test_context_manager(self, model_config):
        """Test async context manager."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        with patch.object(
            SentimentAnalyzer, "initialize", new_callable=AsyncMock
        ) as mock_init:
            with patch.object(
                SentimentAnalyzer, "close", new_callable=AsyncMock
            ) as mock_close:
                async with SentimentAnalyzer(config=model_config) as analyzer:
                    mock_init.assert_called_once()

                mock_close.assert_called_once()

    def test_model_name_property(self, model_config):
        """Test model_name property."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        assert analyzer.model_name == model_config.sentiment_model

    def test_is_initialized_property(self, model_config):
        """Test is_initialized property."""
        from sentimatrix.analysis.sentiment import SentimentAnalyzer

        analyzer = SentimentAnalyzer(config=model_config)

        assert analyzer.is_initialized is False

        analyzer._initialized = True

        assert analyzer.is_initialized is True


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_analyze_sentiment_function(self):
        """Test analyze_sentiment convenience function."""
        from sentimatrix.analysis.sentiment import (
            SentimentAnalyzer,
            SentimentLabel,
            analyze_sentiment,
        )
        from sentimatrix.providers.base import PredictionResult

        mock_provider = AsyncMock()
        mock_provider.initialize = AsyncMock()
        mock_provider.close = AsyncMock()
        mock_provider.predict = AsyncMock(
            return_value=PredictionResult(
                label="positive",
                score=0.95,
                confidence=0.95,
                all_scores={},
                model_name="test",
                processing_time_ms=10.0,
            )
        )

        with patch(
            "sentimatrix.analysis.sentiment.SentimentModelProvider",
            return_value=mock_provider,
        ):
            result = await analyze_sentiment("Great product!")

            assert result.sentiment == SentimentLabel.POSITIVE
            assert result.confidence == 0.95
