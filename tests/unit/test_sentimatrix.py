"""
Unit tests for the main Sentimatrix class.

Tests cover:
- Initialization and configuration
- Sentiment analysis methods
- Emotion detection methods
- Combined analysis methods
- LLM integration (summarize, insights, comparison)
- Pipeline integration
- Platform detection and identifier extraction
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from datetime import datetime

from sentimatrix.main import (
    Sentimatrix,
    AnalysisResult,
    ReviewAnalysisResult,
    InsightsResult,
    ComparisonResult,
    create_sentimatrix,
)
from sentimatrix.core.config import SentimatrixConfig, LLMConfig
from sentimatrix.core.exceptions import (
    SentimatrixError,
    ConfigurationError,
    ValidationError,
)
from sentimatrix.providers.base import Review


class TestSentimatrixConfig:
    """Test Sentimatrix configuration."""

    def test_default_config(self):
        """Test creating Sentimatrix with default config."""
        sm = Sentimatrix()

        assert sm.config is not None
        assert isinstance(sm.config, SentimatrixConfig)
        assert not sm.is_initialized

    def test_dict_config(self):
        """Test creating Sentimatrix with dict config."""
        config = {
            "llm": {
                "provider": "openai",
                "api_key": "test-key",
            }
        }
        sm = Sentimatrix(config=config)

        assert sm.config.llm.api_key == "test-key"

    def test_config_overrides(self):
        """Test config overrides."""
        llm_config = LLMConfig(provider="groq", api_key="groq-key")
        sm = Sentimatrix(llm_config=llm_config)

        assert sm.config.llm.api_key == "groq-key"

    def test_create_sentimatrix_helper(self):
        """Test create_sentimatrix helper function."""
        sm = create_sentimatrix({"llm": {"api_key": "test"}})

        assert isinstance(sm, Sentimatrix)
        assert sm.config.llm.api_key == "test"


class TestSentimatrixInitialization:
    """Test Sentimatrix initialization and cleanup."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test initialization."""
        sm = Sentimatrix()

        with patch.object(sm, '_sentiment_analyzer', None):
            with patch.object(sm, '_emotion_detector', None):
                # Mock the SentimentAnalyzer and EmotionDetector
                with patch('sentimatrix.main.SentimentAnalyzer') as MockSentiment:
                    with patch('sentimatrix.main.EmotionDetector') as MockEmotion:
                        mock_sentiment = AsyncMock()
                        mock_emotion = AsyncMock()
                        MockSentiment.return_value = mock_sentiment
                        MockEmotion.return_value = mock_emotion

                        await sm.initialize()

                        assert sm.is_initialized
                        mock_sentiment.initialize.assert_called_once()
                        mock_emotion.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_double_initialize(self):
        """Test that double initialization is safe."""
        sm = Sentimatrix()
        sm._initialized = True
        sm._sentiment_analyzer = MagicMock()
        sm._emotion_detector = MagicMock()

        # Should return early without error
        await sm.initialize()
        assert sm.is_initialized

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing Sentimatrix."""
        sm = Sentimatrix()
        sm._initialized = True
        sm._sentiment_analyzer = AsyncMock()
        sm._emotion_detector = AsyncMock()

        await sm.close()

        assert not sm.is_initialized
        assert sm._sentiment_analyzer is None
        assert sm._emotion_detector is None

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with patch('sentimatrix.main.SentimentAnalyzer') as MockSentiment:
            with patch('sentimatrix.main.EmotionDetector') as MockEmotion:
                mock_sentiment = AsyncMock()
                mock_emotion = AsyncMock()
                MockSentiment.return_value = mock_sentiment
                MockEmotion.return_value = mock_emotion

                async with Sentimatrix() as sm:
                    assert sm.is_initialized

                # After exit, should be closed
                assert not sm.is_initialized

    def test_ensure_initialized_raises(self):
        """Test that methods raise when not initialized."""
        sm = Sentimatrix()

        with pytest.raises(SentimatrixError, match="not initialized"):
            sm._ensure_initialized()


class TestSentimentAnalysisMethods:
    """Test sentiment analysis methods."""

    @pytest.fixture
    def mock_sentimatrix(self):
        """Create a mock initialized Sentimatrix."""
        sm = Sentimatrix()
        sm._initialized = True
        sm._sentiment_analyzer = AsyncMock()
        sm._emotion_detector = AsyncMock()
        return sm

    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, mock_sentimatrix):
        """Test analyze_sentiment method."""
        mock_result = MagicMock()
        mock_result.sentiment = "positive"
        mock_result.confidence = 0.95
        mock_sentimatrix._sentiment_analyzer.analyze.return_value = mock_result

        result = await mock_sentimatrix.analyze_sentiment("Great product!")

        mock_sentimatrix._sentiment_analyzer.analyze.assert_called_once_with(
            "Great product!", True
        )
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_analyze_sentiment_batch(self, mock_sentimatrix):
        """Test analyze_sentiment_batch method."""
        mock_result = MagicMock()
        mock_result.positive_count = 2
        mock_sentimatrix._sentiment_analyzer.analyze_batch.return_value = mock_result

        result = await mock_sentimatrix.analyze_sentiment_batch(["Good", "Great"])

        mock_sentimatrix._sentiment_analyzer.analyze_batch.assert_called_once()
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_get_quick_sentiment(self, mock_sentimatrix):
        """Test get_quick_sentiment method."""
        mock_sentimatrix._sentiment_analyzer.get_quick_sentiment.return_value = (
            "positive", 0.95
        )

        label, score = await mock_sentimatrix.get_quick_sentiment("Great!")

        assert label == "positive"
        assert score == 0.95


class TestEmotionDetectionMethods:
    """Test emotion detection methods."""

    @pytest.fixture
    def mock_sentimatrix(self):
        """Create a mock initialized Sentimatrix."""
        sm = Sentimatrix()
        sm._initialized = True
        sm._sentiment_analyzer = AsyncMock()
        sm._emotion_detector = AsyncMock()
        return sm

    @pytest.mark.asyncio
    async def test_detect_emotions(self, mock_sentimatrix):
        """Test detect_emotions method."""
        mock_result = MagicMock()
        mock_result.primary_emotion.label = "joy"
        mock_sentimatrix._emotion_detector.detect.return_value = mock_result

        result = await mock_sentimatrix.detect_emotions("I'm so happy!")

        mock_sentimatrix._emotion_detector.detect.assert_called_once()
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_detect_emotions_batch(self, mock_sentimatrix):
        """Test detect_emotions_batch method."""
        mock_result = MagicMock()
        mock_sentimatrix._emotion_detector.detect_batch.return_value = mock_result

        result = await mock_sentimatrix.detect_emotions_batch(["Happy", "Sad"])

        mock_sentimatrix._emotion_detector.detect_batch.assert_called_once()
        assert result == mock_result

    @pytest.mark.asyncio
    async def test_detect_ekman_emotions(self, mock_sentimatrix):
        """Test detect_ekman_emotions method."""
        mock_sentimatrix._emotion_detector.detect_ekman.return_value = {
            "joy": 0.95,
            "anger": 0.01,
        }

        result = await mock_sentimatrix.detect_ekman_emotions("Happy day!")

        assert result["joy"] == 0.95


class TestCombinedAnalysisMethods:
    """Test combined analysis methods."""

    @pytest.fixture
    def mock_sentimatrix(self):
        """Create a mock initialized Sentimatrix."""
        sm = Sentimatrix()
        sm._initialized = True
        sm._sentiment_analyzer = AsyncMock()
        sm._emotion_detector = AsyncMock()
        return sm

    @pytest.mark.asyncio
    async def test_analyze(self, mock_sentimatrix):
        """Test analyze method."""
        mock_sentiment = MagicMock()
        mock_sentiment.is_positive = True
        mock_emotion = MagicMock()
        mock_emotion.primary_emotion.label = "joy"

        mock_sentimatrix._sentiment_analyzer.analyze.return_value = mock_sentiment
        mock_sentimatrix._emotion_detector.detect.return_value = mock_emotion

        result = await mock_sentimatrix.analyze("I love this!")

        assert isinstance(result, AnalysisResult)
        assert result.sentiment == mock_sentiment
        assert result.emotions == mock_emotion

    @pytest.mark.asyncio
    async def test_analyze_without_emotions(self, mock_sentimatrix):
        """Test analyze method without emotions."""
        mock_sentiment = MagicMock()
        mock_sentimatrix._sentiment_analyzer.analyze.return_value = mock_sentiment

        result = await mock_sentimatrix.analyze("Test", include_emotions=False)

        assert result.sentiment == mock_sentiment
        assert result.emotions is None
        mock_sentimatrix._emotion_detector.detect.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_reviews(self, mock_sentimatrix):
        """Test analyze_reviews method."""
        # Create mock reviews
        reviews = [
            Review(id="1", text="Great product!", source="test_source", platform="test"),
            Review(id="2", text="Not good", source="test_source", platform="test"),
        ]

        # Mock batch results
        mock_sentiment_batch = MagicMock()
        mock_sentiment_batch.results = [MagicMock(), MagicMock()]
        mock_sentiment_batch.get_summary.return_value = {"positive_count": 1}

        mock_emotion_batch = MagicMock()
        mock_emotion_batch.results = [MagicMock(), MagicMock()]
        mock_emotion_batch.get_summary.return_value = {"joy": 0.5}

        mock_sentimatrix._sentiment_analyzer.analyze_batch.return_value = mock_sentiment_batch
        mock_sentimatrix._emotion_detector.detect_batch.return_value = mock_emotion_batch

        result = await mock_sentimatrix.analyze_reviews(reviews)

        assert isinstance(result, ReviewAnalysisResult)
        assert result.total_count == 2
        assert len(result.reviews) == 2

    @pytest.mark.asyncio
    async def test_analyze_reviews_empty(self, mock_sentimatrix):
        """Test analyze_reviews with empty list."""
        result = await mock_sentimatrix.analyze_reviews([])

        assert result.total_count == 0
        assert result.reviews == []


class TestPlatformDetection:
    """Test platform detection and ID extraction."""

    def test_detect_amazon_url(self):
        """Test detecting Amazon URL."""
        sm = Sentimatrix()

        assert sm._detect_platform("https://amazon.com/dp/B08N5WRWNW") == "amazon"
        assert sm._detect_platform("https://www.amazon.co.uk/product") == "amazon"

    def test_detect_steam_url(self):
        """Test detecting Steam URL."""
        sm = Sentimatrix()

        assert sm._detect_platform("https://store.steampowered.com/app/730") == "steam"
        assert sm._detect_platform("https://steamcommunity.com/app/570") == "steam"

    def test_detect_youtube_url(self):
        """Test detecting YouTube URL."""
        sm = Sentimatrix()

        assert sm._detect_platform("https://youtube.com/watch?v=dQw4w9WgXcQ") == "youtube"
        assert sm._detect_platform("https://youtu.be/dQw4w9WgXcQ") == "youtube"

    def test_detect_reddit_url(self):
        """Test detecting Reddit URL."""
        sm = Sentimatrix()

        assert sm._detect_platform("https://reddit.com/r/test/comments/abc") == "reddit"
        assert sm._detect_platform("https://redd.it/abc123") == "reddit"

    def test_detect_asin(self):
        """Test detecting ASIN (10 char alphanumeric)."""
        sm = Sentimatrix()

        assert sm._detect_platform("B08N5WRWNW") == "amazon"

    def test_detect_steam_id(self):
        """Test detecting Steam app ID."""
        sm = Sentimatrix()

        assert sm._detect_platform("730") == "steam"

    def test_detect_youtube_id(self):
        """Test detecting YouTube video ID."""
        sm = Sentimatrix()

        assert sm._detect_platform("dQw4w9WgXcQ") == "youtube"

    def test_detect_unknown_raises(self):
        """Test that unknown platform raises error."""
        sm = Sentimatrix()

        with pytest.raises(ValidationError, match="Cannot detect platform"):
            sm._detect_platform("unknown-format-string")

    def test_extract_amazon_identifier(self):
        """Test extracting ASIN from Amazon URL."""
        sm = Sentimatrix()

        assert sm._extract_identifier(
            "https://amazon.com/dp/B08N5WRWNW",
            "amazon"
        ) == "B08N5WRWNW"

        assert sm._extract_identifier(
            "https://amazon.com/gp/product/B08N5WRWNW",
            "amazon"
        ) == "B08N5WRWNW"

    def test_extract_steam_identifier(self):
        """Test extracting app ID from Steam URL."""
        sm = Sentimatrix()

        assert sm._extract_identifier(
            "https://store.steampowered.com/app/730/CSGO",
            "steam"
        ) == "730"

    def test_extract_youtube_identifier(self):
        """Test extracting video ID from YouTube URL."""
        sm = Sentimatrix()

        assert sm._extract_identifier(
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "youtube"
        ) == "dQw4w9WgXcQ"

        assert sm._extract_identifier(
            "https://youtu.be/dQw4w9WgXcQ",
            "youtube"
        ) == "dQw4w9WgXcQ"

    def test_extract_reddit_identifier(self):
        """Test extracting post ID from Reddit URL."""
        sm = Sentimatrix()

        assert sm._extract_identifier(
            "https://reddit.com/r/test/comments/abc123/title",
            "reddit"
        ) == "abc123"


class TestLLMIntegration:
    """Test LLM integration methods."""

    @pytest.fixture
    def mock_sentimatrix_with_llm(self):
        """Create a mock Sentimatrix with LLM configured."""
        config = SentimatrixConfig(
            llm=LLMConfig(provider="openai", api_key="test-key")
        )
        sm = Sentimatrix(config=config)
        sm._initialized = True
        sm._sentiment_analyzer = AsyncMock()
        sm._emotion_detector = AsyncMock()
        return sm

    @pytest.mark.asyncio
    async def test_summarize_reviews_no_llm(self):
        """Test summarize_reviews without LLM config raises error."""
        sm = Sentimatrix()
        sm._initialized = True

        with pytest.raises(ConfigurationError, match="LLM provider not configured"):
            await sm.summarize_reviews([])

    @pytest.mark.asyncio
    async def test_summarize_reviews(self, mock_sentimatrix_with_llm):
        """Test summarize_reviews method."""
        reviews = [
            Review(id="1", text="Great!", source="test", rating=5.0, platform="test"),
            Review(id="2", text="Good", source="test", rating=4.0, platform="test"),
        ]

        mock_manager = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Overall positive reviews."
        mock_manager.generate.return_value = mock_response

        with patch.object(
            mock_sentimatrix_with_llm,
            '_get_llm_manager',
            return_value=mock_manager
        ):
            result = await mock_sentimatrix_with_llm.summarize_reviews(reviews)

        assert result == "Overall positive reviews."
        mock_manager.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_insights(self, mock_sentimatrix_with_llm):
        """Test generate_insights method."""
        reviews = [
            Review(id="1", text="Great product!", source="test", rating=5.0, platform="test"),
        ]

        # Mock analysis
        mock_analysis = MagicMock()
        mock_analysis.total_count = 1
        mock_analysis.positive_ratio = 1.0
        mock_analysis.negative_ratio = 0.0
        mock_analysis.average_polarity = 0.5

        mock_sentimatrix_with_llm.analyze_reviews = AsyncMock(return_value=mock_analysis)

        mock_manager = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = """
SUMMARY: Great product overall.

KEY POINTS:
- High quality
- Good value

PROS:
- Excellent quality
- Fast shipping

CONS:
- Expensive

RECOMMENDATIONS:
- Buy it

THEMES:
- Quality
"""
        mock_manager.generate.return_value = mock_response

        with patch.object(
            mock_sentimatrix_with_llm,
            '_get_llm_manager',
            return_value=mock_manager
        ):
            result = await mock_sentimatrix_with_llm.generate_insights(reviews)

        assert isinstance(result, InsightsResult)
        assert "Great product" in result.summary
        assert len(result.pros) > 0
        assert len(result.cons) > 0

    @pytest.mark.asyncio
    async def test_compare_products(self, mock_sentimatrix_with_llm):
        """Test compare_products method."""
        reviews_a = [
            Review(id="1", text="Great!", source="test", rating=5.0, platform="test"),
        ]
        reviews_b = [
            Review(id="2", text="Okay", source="test", rating=3.0, platform="test"),
        ]

        # Mock analysis results
        mock_analysis_a = MagicMock()
        mock_analysis_a.positive_ratio = 1.0
        mock_analysis_a.negative_ratio = 0.0
        mock_analysis_a.average_polarity = 0.5
        mock_analysis_a.total_count = 1

        mock_analysis_b = MagicMock()
        mock_analysis_b.positive_ratio = 0.5
        mock_analysis_b.negative_ratio = 0.0
        mock_analysis_b.average_polarity = 0.0
        mock_analysis_b.total_count = 1

        mock_sentimatrix_with_llm.analyze_reviews = AsyncMock(
            side_effect=[mock_analysis_a, mock_analysis_b]
        )

        mock_manager = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Product A is better."
        mock_manager.generate.return_value = mock_response

        with patch.object(
            mock_sentimatrix_with_llm,
            '_get_llm_manager',
            return_value=mock_manager
        ):
            result = await mock_sentimatrix_with_llm.compare_products(
                reviews_a, reviews_b,
                "Product A", "Product B"
            )

        assert isinstance(result, ComparisonResult)
        assert result.item_a == "Product A"
        assert result.item_b == "Product B"
        assert result.winner == "Product A"


class TestDataclasses:
    """Test result dataclasses."""

    def test_analysis_result_to_dict(self):
        """Test AnalysisResult to_dict."""
        result = AnalysisResult(
            text="Test",
            sentiment=None,
            emotions=None,
            metadata={"key": "value"},
        )

        data = result.to_dict()

        assert data["text"] == "Test"
        assert data["metadata"] == {"key": "value"}

    def test_review_analysis_result_ratios(self):
        """Test ReviewAnalysisResult ratio calculations."""
        mock_sentiment_pos = MagicMock()
        mock_sentiment_pos.is_positive = True
        mock_sentiment_pos.is_negative = False
        mock_sentiment_pos.polarity = 0.5

        mock_sentiment_neg = MagicMock()
        mock_sentiment_neg.is_positive = False
        mock_sentiment_neg.is_negative = True
        mock_sentiment_neg.polarity = -0.5

        result = ReviewAnalysisResult(
            reviews=[
                AnalysisResult(text="Good", sentiment=mock_sentiment_pos),
                AnalysisResult(text="Bad", sentiment=mock_sentiment_neg),
            ]
        )

        assert result.total_count == 2
        assert result.positive_ratio == 0.5
        assert result.negative_ratio == 0.5
        assert result.average_polarity == 0.0

    def test_insights_result_to_dict(self):
        """Test InsightsResult to_dict."""
        result = InsightsResult(
            summary="Test summary",
            pros=["Pro 1", "Pro 2"],
            cons=["Con 1"],
            key_points=["Point 1"],
            recommendations=["Rec 1"],
            themes=["Theme 1"],
        )

        data = result.to_dict()

        assert data["summary"] == "Test summary"
        assert len(data["pros"]) == 2
        assert len(data["cons"]) == 1

    def test_comparison_result_to_dict(self):
        """Test ComparisonResult to_dict."""
        mock_analysis = MagicMock()
        mock_analysis.positive_ratio = 0.8
        mock_analysis.negative_ratio = 0.1
        mock_analysis.average_polarity = 0.5
        mock_analysis.total_count = 10

        result = ComparisonResult(
            item_a="A",
            item_b="B",
            item_a_analysis=mock_analysis,
            item_b_analysis=mock_analysis,
            winner="A",
        )

        data = result.to_dict()

        assert data["item_a"] == "A"
        assert data["winner"] == "A"
        assert data["item_a_stats"]["positive_ratio"] == 0.8


class TestPipelineMethods:
    """Test pipeline integration methods."""

    @pytest.fixture
    def mock_sentimatrix(self):
        """Create a mock initialized Sentimatrix."""
        sm = Sentimatrix()
        sm._initialized = True
        sm._sentiment_analyzer = AsyncMock()
        sm._emotion_detector = AsyncMock()
        return sm

    @pytest.mark.asyncio
    async def test_run_analysis_pipeline(self, mock_sentimatrix):
        """Test run_analysis_pipeline method."""
        mock_reviews = [
            Review(id="1", text="Great!", source="amazon_scraper", platform="amazon"),
        ]

        mock_analysis = MagicMock(spec=ReviewAnalysisResult)
        mock_analysis.to_dict.return_value = {"total_count": 1}
        mock_analysis.positive_ratio = 1.0
        mock_analysis.negative_ratio = 0.0
        mock_analysis.average_polarity = 0.5
        mock_analysis.total_count = 1

        mock_sentimatrix.scrape_amazon = AsyncMock(return_value=mock_reviews)
        mock_sentimatrix.analyze_reviews = AsyncMock(return_value=mock_analysis)

        result = await mock_sentimatrix.run_analysis_pipeline(
            "B08N5WRWNW",
            platform="amazon",
            limit=10,
            include_insights=False,
        )

        # Pipeline result contains analysis in some format
        assert "pipeline_duration_ms" in result
        mock_sentimatrix.scrape_amazon.assert_called_once()


class TestScraperIntegration:
    """Test scraper integration methods."""

    @pytest.mark.asyncio
    async def test_lazy_scraper_initialization(self):
        """Test that scrapers are lazily initialized."""
        sm = Sentimatrix()

        assert sm._amazon_scraper is None
        assert sm._steam_scraper is None
        assert sm._youtube_scraper is None
        assert sm._reddit_scraper is None

    @pytest.mark.asyncio
    async def test_get_amazon_scraper(self):
        """Test _get_amazon_scraper creates scraper on first call."""
        sm = Sentimatrix()

        with patch('sentimatrix.providers.scrapers.platforms.AmazonScraper') as MockScraper:
            mock_instance = AsyncMock()
            MockScraper.return_value = mock_instance

            # First call should create
            scraper = await sm._get_amazon_scraper()
            assert scraper == mock_instance
            mock_instance.initialize.assert_called_once()

            # Second call should reuse
            scraper2 = await sm._get_amazon_scraper()
            assert scraper2 == mock_instance
            assert mock_instance.initialize.call_count == 1
