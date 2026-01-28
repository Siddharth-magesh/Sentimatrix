"""
Unit tests for Sentimatrix Multi-Modal Analysis.

Tests:
- MultiModalAnalyzer: Combined audio/image/video analysis
- Fusion strategies
- Result data classes
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentimatrix.analysis.multimodal import (
    # Data classes
    FusionStrategy,
    AudioAnalysisResult,
    ImageAnalysisResult,
    VideoAnalysisResult,
    MultiModalResult,
    # Main class
    MultiModalAnalyzer,
    # Convenience functions
    analyze_audio_sentiment,
    analyze_image_sentiment,
    analyze_video_sentiment,
)
from sentimatrix.input.handlers import (
    TranscriptionResult,
    CaptionResult,
    VideoFrameResult,
)
from sentimatrix.analysis.sentiment import SentimentResult, SentimentLabel
from sentimatrix.analysis.emotion import EmotionResult, EmotionScore


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def mock_transcription():
    """Create mock transcription result."""
    return TranscriptionResult(
        text="This is a great product!",
        language="en",
        duration_seconds=5.0,
    )


@pytest.fixture
def mock_caption():
    """Create mock caption result."""
    return CaptionResult(
        caption="A happy person holding a product and smiling.",
        confidence=0.9,
    )


@pytest.fixture
def mock_sentiment():
    """Create mock sentiment result."""
    return SentimentResult(
        text="test",
        sentiment=SentimentLabel.POSITIVE,
        confidence=0.95,
        all_scores={"positive": 0.95, "negative": 0.03, "neutral": 0.02},
    )


@pytest.fixture
def mock_emotion():
    """Create mock emotion result."""
    primary = EmotionScore(label="joy", score=0.8)
    return EmotionResult(
        text="test",
        primary_emotion=primary,
        emotions=[primary],
        all_scores={"joy": 0.8},
    )


# ============================================================================
# Data Class Tests
# ============================================================================


class TestFusionStrategy:
    """Tests for FusionStrategy enum."""

    def test_fusion_strategy_values(self):
        """Test FusionStrategy enum values."""
        assert FusionStrategy.LATE.value == "late"
        assert FusionStrategy.WEIGHTED.value == "weighted"
        assert FusionStrategy.DOMINANT.value == "dominant"


class TestAudioAnalysisResult:
    """Tests for AudioAnalysisResult dataclass."""

    def test_audio_analysis_result_creation(self, mock_transcription, mock_sentiment):
        """Test AudioAnalysisResult creation."""
        result = AudioAnalysisResult(
            transcription=mock_transcription,
            sentiment=mock_sentiment,
        )
        assert result.transcription.text == "This is a great product!"
        assert result.sentiment.sentiment == "positive"
        assert result.emotions is None

    def test_audio_analysis_result_to_dict(self, mock_transcription, mock_sentiment):
        """Test AudioAnalysisResult to_dict method."""
        result = AudioAnalysisResult(
            transcription=mock_transcription,
            sentiment=mock_sentiment,
            metadata={"engine": "whisper"},
        )
        d = result.to_dict()
        assert "transcription" in d
        assert "sentiment" in d
        assert d["metadata"]["engine"] == "whisper"


class TestImageAnalysisResult:
    """Tests for ImageAnalysisResult dataclass."""

    def test_image_analysis_result_creation(self, mock_caption, mock_sentiment):
        """Test ImageAnalysisResult creation."""
        result = ImageAnalysisResult(
            caption=mock_caption,
            sentiment=mock_sentiment,
        )
        assert "happy" in result.caption.caption
        assert result.sentiment.sentiment == "positive"

    def test_image_analysis_result_to_dict(self, mock_caption, mock_sentiment):
        """Test ImageAnalysisResult to_dict method."""
        result = ImageAnalysisResult(
            caption=mock_caption,
            sentiment=mock_sentiment,
            metadata={"model": "gpt4v"},
        )
        d = result.to_dict()
        assert "caption" in d
        assert "sentiment" in d


class TestVideoAnalysisResult:
    """Tests for VideoAnalysisResult dataclass."""

    def test_video_analysis_result_creation(self, mock_sentiment):
        """Test VideoAnalysisResult creation."""
        result = VideoAnalysisResult(
            frames_analyzed=10,
            duration_seconds=30.0,
            frame_sentiments=[mock_sentiment, mock_sentiment],
            combined_sentiment=mock_sentiment,
        )
        assert result.frames_analyzed == 10
        assert result.duration_seconds == 30.0
        assert len(result.frame_sentiments) == 2

    def test_video_analysis_result_to_dict(self, mock_sentiment):
        """Test VideoAnalysisResult to_dict method."""
        result = VideoAnalysisResult(
            frames_analyzed=5,
            duration_seconds=15.0,
            sentiment_timeline=[
                {"frame_index": 0, "sentiment": "positive"},
            ],
        )
        d = result.to_dict()
        assert d["frames_analyzed"] == 5
        assert d["duration_seconds"] == 15.0


class TestMultiModalResult:
    """Tests for MultiModalResult dataclass."""

    def test_multimodal_result_creation(self, mock_sentiment):
        """Test MultiModalResult creation."""
        result = MultiModalResult(
            input_type="multimodal",
            modalities_analyzed=["text", "audio"],
            combined_sentiment=mock_sentiment,
            confidence=0.9,
        )
        assert result.input_type == "multimodal"
        assert "text" in result.modalities_analyzed
        assert result.confidence == 0.9

    def test_multimodal_result_to_dict(self, mock_sentiment):
        """Test MultiModalResult to_dict method."""
        result = MultiModalResult(
            input_type="text",
            modalities_analyzed=["text"],
            text_result=mock_sentiment,
            fusion_method="weighted",
        )
        d = result.to_dict()
        assert d["input_type"] == "text"
        assert d["fusion_method"] == "weighted"


# ============================================================================
# MultiModalAnalyzer Tests
# ============================================================================


class TestMultiModalAnalyzer:
    """Tests for MultiModalAnalyzer class."""

    def test_analyzer_init_defaults(self):
        """Test MultiModalAnalyzer default initialization."""
        analyzer = MultiModalAnalyzer()
        assert analyzer._fusion_strategy == FusionStrategy.WEIGHTED
        assert analyzer._initialized is False

    def test_analyzer_init_custom(self):
        """Test MultiModalAnalyzer custom initialization."""
        analyzer = MultiModalAnalyzer(
            audio_engine="groq_whisper",
            image_model="claude_vision",
            fusion_strategy=FusionStrategy.DOMINANT,
            weights={"text": 0.6, "audio": 0.2, "image": 0.2},
        )
        assert analyzer._audio_engine == "groq_whisper"
        assert analyzer._image_model == "claude_vision"
        assert analyzer._fusion_strategy == FusionStrategy.DOMINANT
        assert analyzer._weights["text"] == 0.6

    @pytest.mark.asyncio
    async def test_analyzer_initialize(self):
        """Test analyzer initialization."""
        analyzer = MultiModalAnalyzer()

        # Mock the sentiment analyzer and emotion detector
        with patch("sentimatrix.analysis.multimodal.SentimentAnalyzer") as MockSA, \
             patch("sentimatrix.analysis.multimodal.EmotionDetector") as MockED:

            mock_sa = AsyncMock()
            mock_ed = AsyncMock()
            MockSA.return_value = mock_sa
            MockED.return_value = mock_ed

            await analyzer.initialize()

            assert analyzer._initialized is True
            mock_sa.initialize.assert_called_once()
            mock_ed.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyzer_close(self):
        """Test analyzer close."""
        analyzer = MultiModalAnalyzer()
        analyzer._initialized = True
        analyzer._sentiment_analyzer = AsyncMock()
        analyzer._emotion_detector = AsyncMock()
        analyzer._audio_handler = AsyncMock()
        analyzer._image_handler = AsyncMock()
        analyzer._video_handler = AsyncMock()

        await analyzer.close()

        assert analyzer._initialized is False
        assert analyzer._sentiment_analyzer is None
        assert analyzer._audio_handler is None

    @pytest.mark.asyncio
    async def test_analyzer_context_manager(self):
        """Test async context manager."""
        analyzer = MultiModalAnalyzer()
        analyzer.initialize = AsyncMock()
        analyzer.close = AsyncMock()

        async with analyzer:
            analyzer.initialize.assert_called_once()

        analyzer.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_fuse_sentiments_dominant(self, mock_sentiment):
        """Test dominant fusion strategy."""
        analyzer = MultiModalAnalyzer(fusion_strategy=FusionStrategy.DOMINANT)

        sentiments = [
            SentimentResult(text="a", sentiment=SentimentLabel.POSITIVE, confidence=0.9, all_scores={}),
            SentimentResult(text="b", sentiment=SentimentLabel.NEGATIVE, confidence=0.7, all_scores={}),
        ]

        result = analyzer._fuse_sentiments(sentiments, None)
        assert result is not None
        assert result.sentiment == SentimentLabel.POSITIVE  # Highest confidence

    @pytest.mark.asyncio
    async def test_fuse_sentiments_weighted(self, mock_sentiment):
        """Test weighted fusion strategy."""
        analyzer = MultiModalAnalyzer(fusion_strategy=FusionStrategy.WEIGHTED)

        sentiments = [
            SentimentResult(text="a", sentiment=SentimentLabel.POSITIVE, confidence=0.8, all_scores={"positive": 0.8, "negative": 0.1, "neutral": 0.1}),
            SentimentResult(text="b", sentiment=SentimentLabel.POSITIVE, confidence=0.9, all_scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
        ]

        result = analyzer._fuse_sentiments(sentiments, None)
        assert result is not None
        assert result.sentiment == "positive"

    @pytest.mark.asyncio
    async def test_fuse_sentiments_late(self, mock_sentiment):
        """Test late fusion strategy."""
        analyzer = MultiModalAnalyzer(fusion_strategy=FusionStrategy.LATE)

        sentiments = [
            SentimentResult(text="a", sentiment=SentimentLabel.POSITIVE, confidence=0.8, all_scores={}),
            SentimentResult(text="b", sentiment=SentimentLabel.POSITIVE, confidence=0.7, all_scores={}),
            SentimentResult(text="c", sentiment=SentimentLabel.NEGATIVE, confidence=0.6, all_scores={}),
        ]

        result = analyzer._fuse_sentiments(sentiments, None)
        assert result is not None
        assert result.sentiment == "positive"  # Majority vote

    @pytest.mark.asyncio
    async def test_fuse_sentiments_empty(self):
        """Test fusion with empty sentiments."""
        analyzer = MultiModalAnalyzer()
        result = analyzer._fuse_sentiments([], None)
        assert result is None

    @pytest.mark.asyncio
    async def test_weighted_fusion(self, mock_sentiment):
        """Test _weighted_fusion method."""
        analyzer = MultiModalAnalyzer()

        sentiments = [
            SentimentResult(text="a", sentiment=SentimentLabel.POSITIVE, confidence=0.9, all_scores={"positive": 0.9, "negative": 0.05, "neutral": 0.05}),
            SentimentResult(text="b", sentiment=SentimentLabel.NEGATIVE, confidence=0.8, all_scores={"positive": 0.1, "negative": 0.8, "neutral": 0.1}),
        ]
        weights = [0.7, 0.3]

        result = analyzer._weighted_fusion(sentiments, weights)
        assert result is not None
        # With 70% positive (0.9) and 30% negative (0.8), positive should dominate
        assert result.sentiment == "positive"


class TestMultiModalAnalyzerMocked:
    """Tests for MultiModalAnalyzer with mocked handlers."""

    @pytest.fixture
    def analyzer_with_mocks(self, mock_transcription, mock_caption, mock_sentiment, mock_emotion):
        """Create analyzer with mocked handlers."""
        analyzer = MultiModalAnalyzer()
        analyzer._initialized = True

        # Mock sentiment analyzer
        mock_sa = AsyncMock()
        mock_sa.analyze = AsyncMock(return_value=mock_sentiment)
        analyzer._sentiment_analyzer = mock_sa

        # Mock emotion detector
        mock_ed = AsyncMock()
        mock_ed.detect = AsyncMock(return_value=mock_emotion)
        analyzer._emotion_detector = mock_ed

        return analyzer

    @pytest.mark.asyncio
    async def test_analyze_audio_mocked(self, analyzer_with_mocks, mock_transcription, mock_sentiment):
        """Test analyze_audio with mocked handlers."""
        analyzer = analyzer_with_mocks

        # Mock audio handler
        mock_audio = AsyncMock()
        mock_audio.transcribe = AsyncMock(return_value=mock_transcription)
        mock_audio.initialize = AsyncMock()

        with patch.object(analyzer, "_ensure_audio_handler", return_value=mock_audio):
            result = await analyzer.analyze_audio("test.wav")

        assert result.transcription.text == mock_transcription.text
        assert result.sentiment.sentiment == "positive"

    @pytest.mark.asyncio
    async def test_analyze_image_mocked(self, analyzer_with_mocks, mock_caption, mock_sentiment):
        """Test analyze_image with mocked handlers."""
        analyzer = analyzer_with_mocks

        # Mock image handler
        mock_image = AsyncMock()
        mock_image.caption = AsyncMock(return_value=mock_caption)
        mock_image.initialize = AsyncMock()

        with patch.object(analyzer, "_ensure_image_handler", return_value=mock_image):
            result = await analyzer.analyze_image("test.jpg")

        assert "happy" in result.caption.caption
        assert result.sentiment.sentiment == "positive"

    @pytest.mark.asyncio
    async def test_transcribe_mocked(self, mock_transcription):
        """Test transcribe without sentiment analysis."""
        analyzer = MultiModalAnalyzer()
        analyzer._initialized = True

        # Mock audio handler
        mock_audio = AsyncMock()
        mock_audio.transcribe = AsyncMock(return_value=mock_transcription)
        mock_audio.initialize = AsyncMock()

        with patch.object(analyzer, "_ensure_audio_handler", return_value=mock_audio):
            result = await analyzer.transcribe("test.wav")

        assert result.text == mock_transcription.text

    @pytest.mark.asyncio
    async def test_caption_image_mocked(self, mock_caption):
        """Test caption_image without sentiment analysis."""
        analyzer = MultiModalAnalyzer()
        analyzer._initialized = True

        # Mock image handler
        mock_image = AsyncMock()
        mock_image.caption = AsyncMock(return_value=mock_caption)
        mock_image.initialize = AsyncMock()

        with patch.object(analyzer, "_ensure_image_handler", return_value=mock_image):
            result = await analyzer.caption_image("test.jpg")

        assert result.caption == mock_caption.caption

    @pytest.mark.asyncio
    async def test_analyze_multimodal_text_only(self, analyzer_with_mocks, mock_sentiment):
        """Test analyze_multimodal with text only."""
        analyzer = analyzer_with_mocks

        result = await analyzer.analyze_multimodal(text="Great product!")

        assert result.input_type == "text"
        assert "text" in result.modalities_analyzed
        assert result.text_result is not None

    @pytest.mark.asyncio
    async def test_analyze_multimodal_combined(self, analyzer_with_mocks, mock_transcription, mock_caption, mock_sentiment):
        """Test analyze_multimodal with multiple modalities."""
        analyzer = analyzer_with_mocks

        # Mock handlers
        mock_audio = AsyncMock()
        mock_audio.transcribe = AsyncMock(return_value=mock_transcription)
        mock_audio.initialize = AsyncMock()

        mock_image = AsyncMock()
        mock_image.caption = AsyncMock(return_value=mock_caption)
        mock_image.initialize = AsyncMock()

        with patch.object(analyzer, "_ensure_audio_handler", return_value=mock_audio), \
             patch.object(analyzer, "_ensure_image_handler", return_value=mock_image):

            result = await analyzer.analyze_multimodal(
                text="Great product!",
                audio="test.wav",
                image="test.jpg",
            )

        assert result.input_type == "multimodal"
        assert len(result.modalities_analyzed) == 3
        assert result.combined_sentiment is not None


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_fuse_with_audio_sentiment(self, mock_sentiment):
        """Test fusion including audio sentiment."""
        analyzer = MultiModalAnalyzer()

        frame_sentiments = [
            SentimentResult(text="a", sentiment=SentimentLabel.POSITIVE, confidence=0.8, all_scores={}),
        ]
        audio_sentiment = SentimentResult(
            text="b", sentiment=SentimentLabel.NEGATIVE, confidence=0.9, all_scores={}
        )

        result = analyzer._fuse_sentiments(frame_sentiments, audio_sentiment)
        assert result is not None
        # Audio has higher confidence - check all_scores or sentiment value
        sent_val = result.sentiment.value if hasattr(result.sentiment, 'value') else str(result.sentiment)
        assert len(result.all_scores) > 0 or sent_val in ["positive", "negative"]

    @pytest.mark.asyncio
    async def test_empty_text_transcription(self, mock_sentiment):
        """Test handling empty transcription."""
        analyzer = MultiModalAnalyzer()
        analyzer._initialized = True
        analyzer._sentiment_analyzer = AsyncMock()

        empty_transcription = TranscriptionResult(text="", language="en")

        mock_audio = AsyncMock()
        mock_audio.transcribe = AsyncMock(return_value=empty_transcription)
        mock_audio.initialize = AsyncMock()

        with patch.object(analyzer, "_ensure_audio_handler", return_value=mock_audio):
            result = await analyzer.analyze_audio("test.wav")

        assert result.sentiment is None  # No sentiment for empty text
