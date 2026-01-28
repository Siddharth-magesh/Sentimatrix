"""
Unit Tests for Emotion Detection Module

Tests the emotion detection functionality including:
- EmotionScore dataclass
- EmotionResult dataclass
- BatchEmotionResult dataclass
- EmotionDetector class
- Emotion mappings (GoEmotions, Ekman)
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentimatrix.core.config import ModelConfig, SentimatrixConfig
from sentimatrix.core.exceptions import InvalidInputError, SentimatrixError, ValidationError


class TestEmotionCategory:
    """Tests for EmotionCategory enum."""

    def test_goemotions_categories_exist(self):
        """Test that all GoEmotions categories are defined."""
        from sentimatrix.analysis.emotion import EmotionCategory

        # Test subset of 28 GoEmotions
        assert EmotionCategory.JOY
        assert EmotionCategory.SADNESS
        assert EmotionCategory.ANGER
        assert EmotionCategory.FEAR
        assert EmotionCategory.SURPRISE
        assert EmotionCategory.DISGUST
        assert EmotionCategory.LOVE
        assert EmotionCategory.NEUTRAL

    def test_goemotions_count(self):
        """Test that we have all 28 GoEmotions."""
        from sentimatrix.analysis.emotion import EmotionCategory

        assert len(EmotionCategory) == 28


class TestEkmanEmotion:
    """Tests for EkmanEmotion enum."""

    def test_ekman_emotions_exist(self):
        """Test that all Ekman emotions are defined."""
        from sentimatrix.analysis.emotion import EkmanEmotion

        assert EkmanEmotion.ANGER
        assert EkmanEmotion.DISGUST
        assert EkmanEmotion.FEAR
        assert EkmanEmotion.JOY
        assert EkmanEmotion.SADNESS
        assert EkmanEmotion.SURPRISE

    def test_ekman_count(self):
        """Test that we have exactly 6 Ekman emotions."""
        from sentimatrix.analysis.emotion import EkmanEmotion

        assert len(EkmanEmotion) == 6


class TestPlutchikEmotion:
    """Tests for PlutchikEmotion enum."""

    def test_plutchik_emotions_exist(self):
        """Test that all Plutchik emotions are defined."""
        from sentimatrix.analysis.emotion import PlutchikEmotion

        assert PlutchikEmotion.ANGER
        assert PlutchikEmotion.ANTICIPATION
        assert PlutchikEmotion.DISGUST
        assert PlutchikEmotion.FEAR
        assert PlutchikEmotion.JOY
        assert PlutchikEmotion.SADNESS
        assert PlutchikEmotion.SURPRISE
        assert PlutchikEmotion.TRUST

    def test_plutchik_count(self):
        """Test that we have exactly 8 Plutchik emotions."""
        from sentimatrix.analysis.emotion import PlutchikEmotion

        assert len(PlutchikEmotion) == 8


class TestEmotionMode:
    """Tests for EmotionMode enum."""

    def test_emotion_modes_exist(self):
        """Test emotion mode values."""
        from sentimatrix.analysis.emotion import EmotionMode

        assert EmotionMode.SINGLE_LABEL.value == "single_label"
        assert EmotionMode.MULTI_LABEL.value == "multi_label"
        assert EmotionMode.TOP_K.value == "top_k"


class TestEmotionMappings:
    """Tests for emotion mappings."""

    def test_goemotions_to_ekman_mapping(self):
        """Test GoEmotions to Ekman mapping."""
        from sentimatrix.analysis.emotion import GOEMOTIONS_TO_EKMAN, EkmanEmotion

        assert GOEMOTIONS_TO_EKMAN["anger"] == EkmanEmotion.ANGER
        assert GOEMOTIONS_TO_EKMAN["joy"] == EkmanEmotion.JOY
        assert GOEMOTIONS_TO_EKMAN["sadness"] == EkmanEmotion.SADNESS
        assert GOEMOTIONS_TO_EKMAN["fear"] == EkmanEmotion.FEAR
        assert GOEMOTIONS_TO_EKMAN["disgust"] == EkmanEmotion.DISGUST
        assert GOEMOTIONS_TO_EKMAN["surprise"] == EkmanEmotion.SURPRISE

    def test_emotion_valence_mapping(self):
        """Test emotion valence mapping."""
        from sentimatrix.analysis.emotion import EMOTION_VALENCE

        # Positive emotions
        assert EMOTION_VALENCE["joy"] == "positive"
        assert EMOTION_VALENCE["love"] == "positive"
        assert EMOTION_VALENCE["gratitude"] == "positive"

        # Negative emotions
        assert EMOTION_VALENCE["anger"] == "negative"
        assert EMOTION_VALENCE["sadness"] == "negative"
        assert EMOTION_VALENCE["fear"] == "negative"

        # Neutral emotions
        assert EMOTION_VALENCE["neutral"] == "neutral"
        assert EMOTION_VALENCE["surprise"] == "neutral"


class TestEmotionScore:
    """Tests for EmotionScore dataclass."""

    def test_create_emotion_score(self):
        """Test creating an emotion score."""
        from sentimatrix.analysis.emotion import EmotionScore

        score = EmotionScore(label="joy", score=0.95)

        assert score.label == "joy"
        assert score.score == 0.95

    def test_emotion_score_post_init(self):
        """Test post-init processing."""
        from sentimatrix.analysis.emotion import EkmanEmotion, EmotionCategory, EmotionScore

        score = EmotionScore(label="joy", score=0.95)

        assert score.category == EmotionCategory.JOY
        assert score.valence == "positive"
        assert score.ekman_mapping == EkmanEmotion.JOY

    def test_emotion_score_unknown_label(self):
        """Test emotion score with unknown label."""
        from sentimatrix.analysis.emotion import EmotionScore

        score = EmotionScore(label="unknown_emotion", score=0.5)

        assert score.category is None
        assert score.valence == "neutral"
        assert score.ekman_mapping is None

    def test_emotion_score_to_dict(self):
        """Test emotion score serialization."""
        from sentimatrix.analysis.emotion import EmotionScore

        score = EmotionScore(label="joy", score=0.95)
        score_dict = score.to_dict()

        assert score_dict["label"] == "joy"
        assert score_dict["score"] == 0.95
        assert score_dict["category"] == "joy"
        assert score_dict["valence"] == "positive"
        assert score_dict["ekman_mapping"] == "joy"


class TestEmotionResult:
    """Tests for EmotionResult dataclass."""

    def test_create_emotion_result(self):
        """Test creating an emotion result."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        primary = EmotionScore(label="joy", score=0.95)
        emotions = [
            primary,
            EmotionScore(label="excitement", score=0.80),
            EmotionScore(label="love", score=0.60),
        ]

        result = EmotionResult(
            text="I'm so happy today!",
            primary_emotion=primary,
            emotions=emotions,
            all_scores={"joy": 0.95, "excitement": 0.80, "love": 0.60},
            model_name="test-model",
            processing_time_ms=15.0,
        )

        assert result.text == "I'm so happy today!"
        assert result.primary_emotion.label == "joy"
        assert len(result.emotions) == 3

    def test_is_positive(self):
        """Test is_positive property."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        primary = EmotionScore(label="joy", score=0.95)
        result = EmotionResult(
            text="Happy!",
            primary_emotion=primary,
            emotions=[primary],
        )

        assert result.is_positive is True
        assert result.is_negative is False
        assert result.is_neutral is False

    def test_is_negative(self):
        """Test is_negative property."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        primary = EmotionScore(label="anger", score=0.90)
        result = EmotionResult(
            text="So angry!",
            primary_emotion=primary,
            emotions=[primary],
        )

        assert result.is_negative is True
        assert result.is_positive is False

    def test_is_neutral(self):
        """Test is_neutral property."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        primary = EmotionScore(label="neutral", score=0.70)
        result = EmotionResult(
            text="Okay.",
            primary_emotion=primary,
            emotions=[primary],
        )

        assert result.is_neutral is True

    def test_ekman_emotion(self):
        """Test ekman_emotion property."""
        from sentimatrix.analysis.emotion import EkmanEmotion, EmotionResult, EmotionScore

        primary = EmotionScore(label="joy", score=0.95)
        result = EmotionResult(
            text="Happy!",
            primary_emotion=primary,
            emotions=[primary],
        )

        assert result.ekman_emotion == EkmanEmotion.JOY

    def test_positive_emotions(self):
        """Test positive_emotions property."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        emotions = [
            EmotionScore(label="joy", score=0.95),
            EmotionScore(label="anger", score=0.30),
            EmotionScore(label="love", score=0.80),
        ]

        result = EmotionResult(
            text="Happy!",
            primary_emotion=emotions[0],
            emotions=emotions,
        )

        positive = result.positive_emotions
        assert len(positive) == 2
        assert all(e.valence == "positive" for e in positive)

    def test_negative_emotions(self):
        """Test negative_emotions property."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        emotions = [
            EmotionScore(label="anger", score=0.90),
            EmotionScore(label="sadness", score=0.70),
            EmotionScore(label="joy", score=0.20),
        ]

        result = EmotionResult(
            text="Angry!",
            primary_emotion=emotions[0],
            emotions=emotions,
        )

        negative = result.negative_emotions
        assert len(negative) == 2
        assert all(e.valence == "negative" for e in negative)

    def test_emotion_labels(self):
        """Test emotion_labels property."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        emotions = [
            EmotionScore(label="joy", score=0.95),
            EmotionScore(label="love", score=0.80),
        ]

        result = EmotionResult(
            text="Happy!",
            primary_emotion=emotions[0],
            emotions=emotions,
        )

        labels = result.emotion_labels
        assert labels == ["joy", "love"]

    def test_get_ekman_distribution(self):
        """Test get_ekman_distribution method."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        result = EmotionResult(
            text="Happy!",
            primary_emotion=EmotionScore(label="joy", score=0.95),
            emotions=[],
            all_scores={"joy": 0.95, "anger": 0.10, "sadness": 0.05},
        )

        distribution = result.get_ekman_distribution()

        assert distribution["joy"] == 0.95
        assert distribution["anger"] == 0.10
        assert distribution["sadness"] == 0.05
        assert "fear" in distribution
        assert "disgust" in distribution
        assert "surprise" in distribution

    def test_to_dict(self):
        """Test result serialization."""
        from sentimatrix.analysis.emotion import EmotionResult, EmotionScore

        primary = EmotionScore(label="joy", score=0.95)
        result = EmotionResult(
            text="Happy!",
            primary_emotion=primary,
            emotions=[primary],
            all_scores={"joy": 0.95},
            model_name="test-model",
            processing_time_ms=10.0,
        )

        result_dict = result.to_dict()

        assert result_dict["text"] == "Happy!"
        assert result_dict["primary_emotion"]["label"] == "joy"
        assert result_dict["is_positive"] is True
        assert "ekman_distribution" in result_dict


class TestBatchEmotionResult:
    """Tests for BatchEmotionResult dataclass."""

    def test_empty_batch_result(self):
        """Test empty batch result."""
        from sentimatrix.analysis.emotion import BatchEmotionResult

        batch = BatchEmotionResult(results=[])

        assert batch.total_count == 0
        assert batch.emotion_counts == {}
        assert batch.valence_counts == {}

    def test_batch_statistics(self):
        """Test batch statistics calculation."""
        from sentimatrix.analysis.emotion import (
            BatchEmotionResult,
            EmotionResult,
            EmotionScore,
        )

        results = [
            EmotionResult(
                text="Happy!",
                primary_emotion=EmotionScore(label="joy", score=0.95),
                emotions=[],
            ),
            EmotionResult(
                text="Angry!",
                primary_emotion=EmotionScore(label="anger", score=0.90),
                emotions=[],
            ),
            EmotionResult(
                text="Also happy!",
                primary_emotion=EmotionScore(label="joy", score=0.85),
                emotions=[],
            ),
        ]

        batch = BatchEmotionResult(results=results)

        assert batch.total_count == 3
        assert batch.emotion_counts["joy"] == 2
        assert batch.emotion_counts["anger"] == 1
        assert batch.valence_counts["positive"] == 2
        assert batch.valence_counts["negative"] == 1

    def test_most_common_emotion(self):
        """Test most_common_emotion property."""
        from sentimatrix.analysis.emotion import (
            BatchEmotionResult,
            EmotionResult,
            EmotionScore,
        )

        results = [
            EmotionResult(
                text="Happy!",
                primary_emotion=EmotionScore(label="joy", score=0.95),
                emotions=[],
            ),
            EmotionResult(
                text="Also happy!",
                primary_emotion=EmotionScore(label="joy", score=0.90),
                emotions=[],
            ),
            EmotionResult(
                text="Angry!",
                primary_emotion=EmotionScore(label="anger", score=0.85),
                emotions=[],
            ),
        ]

        batch = BatchEmotionResult(results=results)
        emotion, count = batch.most_common_emotion

        assert emotion == "joy"
        assert count == 2

    def test_positive_negative_ratios(self):
        """Test positive and negative ratios."""
        from sentimatrix.analysis.emotion import (
            BatchEmotionResult,
            EmotionResult,
            EmotionScore,
        )

        results = [
            EmotionResult(
                text="Happy!",
                primary_emotion=EmotionScore(label="joy", score=0.95),
                emotions=[],
            ),
            EmotionResult(
                text="Angry!",
                primary_emotion=EmotionScore(label="anger", score=0.90),
                emotions=[],
            ),
        ]

        batch = BatchEmotionResult(results=results)

        assert batch.positive_ratio == 0.5
        assert batch.negative_ratio == 0.5

    def test_get_emotion_distribution(self):
        """Test emotion distribution calculation."""
        from sentimatrix.analysis.emotion import (
            BatchEmotionResult,
            EmotionResult,
            EmotionScore,
        )

        results = [
            EmotionResult(
                text="Happy!",
                primary_emotion=EmotionScore(label="joy", score=0.95),
                emotions=[],
            ),
            EmotionResult(
                text="Also happy!",
                primary_emotion=EmotionScore(label="joy", score=0.90),
                emotions=[],
            ),
        ]

        batch = BatchEmotionResult(results=results)
        distribution = batch.get_emotion_distribution()

        assert distribution["joy"] == 1.0  # 2/2

    def test_get_summary(self):
        """Test getting summary without results."""
        from sentimatrix.analysis.emotion import (
            BatchEmotionResult,
            EmotionResult,
            EmotionScore,
        )

        results = [
            EmotionResult(
                text="Happy!",
                primary_emotion=EmotionScore(label="joy", score=0.95),
                emotions=[],
            ),
        ]

        batch = BatchEmotionResult(results=results)
        summary = batch.get_summary()

        assert "results" not in summary
        assert summary["total_count"] == 1
        assert summary["most_common_emotion"] == "joy"


class TestEmotionDetector:
    """Tests for EmotionDetector class."""

    @pytest.fixture
    def model_config(self) -> ModelConfig:
        """Create model configuration."""
        return ModelConfig(
            emotion_model="SamLowe/roberta-base-go_emotions",
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
                label="joy",
                score=0.95,
                confidence=0.95,
                all_scores={
                    "joy": 0.95,
                    "excitement": 0.80,
                    "love": 0.60,
                    "anger": 0.05,
                    "sadness": 0.02,
                },
                model_name="test-model",
                processing_time_ms=10.0,
            )
        )
        provider.predict_batch = AsyncMock(
            return_value=[
                PredictionResult(
                    label="joy",
                    score=0.95,
                    confidence=0.95,
                    all_scores={"joy": 0.95, "anger": 0.05},
                    model_name="test-model",
                    processing_time_ms=5.0,
                ),
                PredictionResult(
                    label="anger",
                    score=0.90,
                    confidence=0.90,
                    all_scores={"anger": 0.90, "joy": 0.10},
                    model_name="test-model",
                    processing_time_ms=5.0,
                ),
            ]
        )
        return provider

    def test_detector_init_with_model_config(self, model_config):
        """Test detector initialization with ModelConfig."""
        from sentimatrix.analysis.emotion import EmotionDetector, EmotionMode

        detector = EmotionDetector(config=model_config)

        assert detector._model_name == model_config.emotion_model
        assert detector._mode == EmotionMode.MULTI_LABEL
        assert not detector.is_initialized

    def test_detector_init_with_sentimatrix_config(self):
        """Test detector initialization with SentimatrixConfig."""
        from sentimatrix.analysis.emotion import EmotionDetector

        config = SentimatrixConfig()
        detector = EmotionDetector(config=config)

        assert detector._model_name == config.models.emotion_model

    def test_detector_init_with_custom_settings(self, model_config):
        """Test detector initialization with custom settings."""
        from sentimatrix.analysis.emotion import EmotionDetector, EmotionMode

        detector = EmotionDetector(
            config=model_config,
            mode=EmotionMode.TOP_K,
            threshold=0.5,
            top_k=5,
        )

        assert detector.mode == EmotionMode.TOP_K
        assert detector.threshold == 0.5
        assert detector.top_k == 5

    def test_validate_text_valid(self, model_config):
        """Test text validation with valid input."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)

        result = detector._validate_text("  Valid text  ")
        assert result == "Valid text"

    def test_validate_text_none_raises_error(self, model_config):
        """Test text validation with None."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)

        with pytest.raises(InvalidInputError):
            detector._validate_text(None)

    def test_validate_text_empty_raises_error(self, model_config):
        """Test text validation with empty string."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)

        with pytest.raises(InvalidInputError):
            detector._validate_text("   ")

    def test_filter_emotions_single_label(self, model_config):
        """Test emotion filtering in single label mode."""
        from sentimatrix.analysis.emotion import EmotionDetector, EmotionMode

        detector = EmotionDetector(config=model_config, mode=EmotionMode.SINGLE_LABEL)

        all_scores = {"joy": 0.95, "anger": 0.30, "sadness": 0.10}
        emotions = detector._filter_emotions(all_scores)

        assert len(emotions) == 1
        assert emotions[0].label == "joy"

    def test_filter_emotions_top_k(self, model_config):
        """Test emotion filtering in top-k mode."""
        from sentimatrix.analysis.emotion import EmotionDetector, EmotionMode

        detector = EmotionDetector(config=model_config, mode=EmotionMode.TOP_K, top_k=2)

        all_scores = {"joy": 0.95, "anger": 0.30, "sadness": 0.10}
        emotions = detector._filter_emotions(all_scores)

        assert len(emotions) == 2
        assert emotions[0].label == "joy"
        assert emotions[1].label == "anger"

    def test_filter_emotions_multi_label(self, model_config):
        """Test emotion filtering in multi-label mode."""
        from sentimatrix.analysis.emotion import EmotionDetector, EmotionMode

        detector = EmotionDetector(
            config=model_config, mode=EmotionMode.MULTI_LABEL, threshold=0.25
        )

        all_scores = {"joy": 0.95, "anger": 0.30, "sadness": 0.10}
        emotions = detector._filter_emotions(all_scores)

        assert len(emotions) == 2  # joy and anger above 0.25
        assert emotions[0].label == "joy"
        assert emotions[1].label == "anger"

    def test_filter_emotions_multi_label_fallback(self, model_config):
        """Test multi-label mode falls back to top emotion if none above threshold."""
        from sentimatrix.analysis.emotion import EmotionDetector, EmotionMode

        detector = EmotionDetector(
            config=model_config, mode=EmotionMode.MULTI_LABEL, threshold=0.99
        )

        all_scores = {"joy": 0.95, "anger": 0.30}
        emotions = detector._filter_emotions(all_scores)

        # Should fall back to at least one emotion
        assert len(emotions) >= 1

    @pytest.mark.asyncio
    async def test_detect_not_initialized_raises_error(self, model_config):
        """Test detect without initialization."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)

        with pytest.raises(SentimatrixError):
            await detector.detect("Test text")

    @pytest.mark.asyncio
    async def test_detect_with_mock_provider(self, model_config, mock_provider):
        """Test detect with mocked provider."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)
        detector._provider = mock_provider
        detector._initialized = True

        result = await detector.detect("I'm so happy today!")

        assert result.primary_emotion.label == "joy"
        assert result.primary_emotion.score == 0.95
        assert result.text == "I'm so happy today!"
        mock_provider.predict.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_with_mode_override(self, model_config, mock_provider):
        """Test detect with mode override."""
        from sentimatrix.analysis.emotion import EmotionDetector, EmotionMode

        detector = EmotionDetector(
            config=model_config, mode=EmotionMode.MULTI_LABEL, threshold=0.3
        )
        detector._provider = mock_provider
        detector._initialized = True

        # Override to single label
        result = await detector.detect(
            "I'm so happy!",
            mode=EmotionMode.SINGLE_LABEL,
        )

        assert len(result.emotions) == 1

    @pytest.mark.asyncio
    async def test_detect_batch_with_mock_provider(self, model_config, mock_provider):
        """Test detect_batch with mocked provider."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)
        detector._provider = mock_provider
        detector._initialized = True

        texts = ["I'm happy!", "I'm angry!"]
        result = await detector.detect_batch(texts)

        assert result.total_count == 2
        assert result.emotion_counts["joy"] == 1
        assert result.emotion_counts["anger"] == 1
        mock_provider.predict_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_batch_empty_list(self, model_config, mock_provider):
        """Test detect_batch with empty list."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)
        detector._provider = mock_provider
        detector._initialized = True

        result = await detector.detect_batch([])

        assert result.total_count == 0
        assert result.results == []

    @pytest.mark.asyncio
    async def test_detect_top_k(self, model_config, mock_provider):
        """Test detect_top_k convenience method."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)
        detector._provider = mock_provider
        detector._initialized = True

        emotions = await detector.detect_top_k("Happy!", k=3)

        assert len(emotions) <= 3
        assert all("label" in e for e in emotions)
        assert all("score" in e for e in emotions)

    @pytest.mark.asyncio
    async def test_detect_multi_label(self, model_config, mock_provider):
        """Test detect_multi_label convenience method."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)
        detector._provider = mock_provider
        detector._initialized = True

        emotions = await detector.detect_multi_label("Happy!", threshold=0.5)

        assert isinstance(emotions, list)

    @pytest.mark.asyncio
    async def test_detect_ekman(self, model_config, mock_provider):
        """Test detect_ekman convenience method."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)
        detector._provider = mock_provider
        detector._initialized = True

        ekman = await detector.detect_ekman("Happy!")

        assert "joy" in ekman
        assert "anger" in ekman
        assert "sadness" in ekman
        assert "fear" in ekman
        assert "disgust" in ekman
        assert "surprise" in ekman

    @pytest.mark.asyncio
    async def test_context_manager(self, model_config):
        """Test async context manager."""
        from sentimatrix.analysis.emotion import EmotionDetector

        with patch.object(
            EmotionDetector, "initialize", new_callable=AsyncMock
        ) as mock_init:
            with patch.object(
                EmotionDetector, "close", new_callable=AsyncMock
            ) as mock_close:
                async with EmotionDetector(config=model_config) as detector:
                    mock_init.assert_called_once()

                mock_close.assert_called_once()

    def test_model_name_property(self, model_config):
        """Test model_name property."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)

        assert detector.model_name == model_config.emotion_model

    def test_mode_property(self, model_config):
        """Test mode property."""
        from sentimatrix.analysis.emotion import EmotionDetector, EmotionMode

        detector = EmotionDetector(config=model_config, mode=EmotionMode.TOP_K)

        assert detector.mode == EmotionMode.TOP_K

    def test_threshold_property(self, model_config):
        """Test threshold property."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config, threshold=0.5)

        assert detector.threshold == 0.5

    def test_top_k_property(self, model_config):
        """Test top_k property."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config, top_k=5)

        assert detector.top_k == 5

    def test_is_initialized_property(self, model_config):
        """Test is_initialized property."""
        from sentimatrix.analysis.emotion import EmotionDetector

        detector = EmotionDetector(config=model_config)

        assert detector.is_initialized is False

        detector._initialized = True

        assert detector.is_initialized is True


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_detect_emotions_function(self):
        """Test detect_emotions convenience function."""
        from sentimatrix.analysis.emotion import EmotionDetector, detect_emotions
        from sentimatrix.providers.base import PredictionResult

        mock_provider = AsyncMock()
        mock_provider.initialize = AsyncMock()
        mock_provider.close = AsyncMock()
        mock_provider.predict = AsyncMock(
            return_value=PredictionResult(
                label="joy",
                score=0.95,
                confidence=0.95,
                all_scores={"joy": 0.95, "anger": 0.05},
                model_name="test",
                processing_time_ms=10.0,
            )
        )

        with patch(
            "sentimatrix.analysis.emotion.EmotionModelProvider",
            return_value=mock_provider,
        ):
            result = await detect_emotions("I'm so happy!")

            assert result.primary_emotion.label == "joy"
            assert result.primary_emotion.score == 0.95
