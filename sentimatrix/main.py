"""
Sentimatrix - Main Entry Point

The primary interface for all Sentimatrix functionality including:
- Sentiment analysis
- Emotion detection
- Web scraping (platform-specific and generic)
- LLM-powered insights and summarization
- Pipeline orchestration

Example:
    >>> from sentimatrix import Sentimatrix
    >>>
    >>> async with Sentimatrix() as sm:
    ...     # Quick sentiment analysis
    ...     result = await sm.analyze_sentiment("I love this product!")
    ...     print(result.sentiment)  # "positive"
    ...
    ...     # Scrape and analyze reviews
    ...     reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)
    ...     analysis = await sm.analyze_reviews(reviews)
    ...
    ...     # Generate insights with LLM
    ...     insights = await sm.generate_insights(reviews)
    ...     print(insights.summary)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from sentimatrix.core.config import (
    SentimatrixConfig,
    LLMConfig,
    ScraperConfig,
    ModelConfig,
)
from sentimatrix.core.exceptions import (
    ConfigurationError,
    ProviderInitializationError,
    SentimatrixError,
    ValidationError,
)
from sentimatrix.core.logger import get_logger
from sentimatrix.core.pipeline import (
    Pipeline,
    PipelineContext,
    PipelineResult,
    FunctionStep,
    ParallelSteps,
)
from sentimatrix.analysis.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    BatchSentimentResult,
    SentimentClass,
)
from sentimatrix.analysis.emotion import (
    EmotionDetector,
    EmotionResult,
    BatchEmotionResult,
    EmotionMode,
)
from sentimatrix.providers.base import Review

if TYPE_CHECKING:
    from sentimatrix.analysis.multimodal import (
        AudioAnalysisResult,
        ImageAnalysisResult,
        VideoAnalysisResult,
        MultiModalResult,
    )
    from sentimatrix.input.handlers import TranscriptionResult, CaptionResult

logger = get_logger(__name__)


@dataclass
class AnalysisResult:
    """
    Combined analysis result for a single text or review.

    Attributes:
        text: Original text
        sentiment: Sentiment analysis result
        emotions: Emotion detection result
        metadata: Additional metadata
    """
    text: str
    sentiment: Optional[SentimentResult] = None
    emotions: Optional[EmotionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "emotions": self.emotions.to_dict() if self.emotions else None,
            "metadata": self.metadata,
        }


@dataclass
class ReviewAnalysisResult:
    """
    Analysis result for a collection of reviews.

    Attributes:
        reviews: List of analyzed reviews
        sentiment_summary: Aggregate sentiment statistics
        emotion_summary: Aggregate emotion statistics
        total_count: Total number of reviews analyzed
    """
    reviews: List[AnalysisResult]
    sentiment_summary: Optional[Dict[str, Any]] = None
    emotion_summary: Optional[Dict[str, Any]] = None
    total_count: int = 0

    def __post_init__(self) -> None:
        """Calculate aggregates."""
        self.total_count = len(self.reviews)

    @property
    def positive_ratio(self) -> float:
        """Get ratio of positive sentiments."""
        if not self.reviews:
            return 0.0
        positive = sum(
            1 for r in self.reviews
            if r.sentiment and r.sentiment.is_positive
        )
        return positive / len(self.reviews)

    @property
    def negative_ratio(self) -> float:
        """Get ratio of negative sentiments."""
        if not self.reviews:
            return 0.0
        negative = sum(
            1 for r in self.reviews
            if r.sentiment and r.sentiment.is_negative
        )
        return negative / len(self.reviews)

    @property
    def average_polarity(self) -> float:
        """Get average polarity score."""
        if not self.reviews:
            return 0.0
        polarities = [
            r.sentiment.polarity
            for r in self.reviews
            if r.sentiment
        ]
        return sum(polarities) / len(polarities) if polarities else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "average_polarity": self.average_polarity,
            "sentiment_summary": self.sentiment_summary,
            "emotion_summary": self.emotion_summary,
            "reviews": [r.to_dict() for r in self.reviews],
        }


@dataclass
class InsightsResult:
    """
    LLM-generated insights for reviews.

    Attributes:
        summary: Overall summary of reviews
        key_points: List of key points extracted
        pros: List of positive aspects
        cons: List of negative aspects
        recommendations: Suggested improvements
        themes: Common themes identified
        raw_response: Raw LLM response
    """
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "key_points": self.key_points,
            "pros": self.pros,
            "cons": self.cons,
            "recommendations": self.recommendations,
            "themes": self.themes,
        }


@dataclass
class ComparisonResult:
    """
    Comparison result between two products/items.

    Attributes:
        item_a: Name/identifier of first item
        item_b: Name/identifier of second item
        item_a_analysis: Analysis for first item
        item_b_analysis: Analysis for second item
        comparison_summary: LLM-generated comparison
        winner: Which item is preferred overall
        differences: Key differences identified
    """
    item_a: str
    item_b: str
    item_a_analysis: Optional[ReviewAnalysisResult] = None
    item_b_analysis: Optional[ReviewAnalysisResult] = None
    comparison_summary: str = ""
    winner: Optional[str] = None
    differences: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "item_a": self.item_a,
            "item_b": self.item_b,
            "item_a_stats": {
                "positive_ratio": self.item_a_analysis.positive_ratio if self.item_a_analysis else 0,
                "negative_ratio": self.item_a_analysis.negative_ratio if self.item_a_analysis else 0,
                "average_polarity": self.item_a_analysis.average_polarity if self.item_a_analysis else 0,
                "total_reviews": self.item_a_analysis.total_count if self.item_a_analysis else 0,
            },
            "item_b_stats": {
                "positive_ratio": self.item_b_analysis.positive_ratio if self.item_b_analysis else 0,
                "negative_ratio": self.item_b_analysis.negative_ratio if self.item_b_analysis else 0,
                "average_polarity": self.item_b_analysis.average_polarity if self.item_b_analysis else 0,
                "total_reviews": self.item_b_analysis.total_count if self.item_b_analysis else 0,
            },
            "comparison_summary": self.comparison_summary,
            "winner": self.winner,
            "differences": self.differences,
        }


class Sentimatrix:
    """
    Main Sentimatrix class providing unified access to all functionality.

    Features:
    - Sentiment analysis (quick, batch, configurable models)
    - Emotion detection (GoEmotions, Ekman mapping)
    - Web scraping (Amazon, Steam, YouTube, Reddit, generic)
    - LLM-powered summarization and insights
    - Pipeline orchestration for complex workflows
    - Comparison between products/items

    Example:
        >>> from sentimatrix import Sentimatrix
        >>>
        >>> # Using async context manager (recommended)
        >>> async with Sentimatrix() as sm:
        ...     result = await sm.analyze_sentiment("Great product!")
        ...     print(result.sentiment)
        ...
        >>> # Manual initialization
        >>> sm = Sentimatrix()
        >>> await sm.initialize()
        >>> result = await sm.analyze_sentiment("Great product!")
        >>> await sm.close()
    """

    def __init__(
        self,
        config: Optional[Union[SentimatrixConfig, Dict[str, Any]]] = None,
        *,
        llm_config: Optional[LLMConfig] = None,
        scraper_config: Optional[ScraperConfig] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        """
        Initialize Sentimatrix.

        Args:
            config: Main configuration (SentimatrixConfig or dict)
            llm_config: Override LLM configuration
            scraper_config: Override scraper configuration
            model_config: Override model configuration
        """
        # Parse configuration
        if isinstance(config, dict):
            self._config = SentimatrixConfig(**config)
        elif config is not None:
            self._config = config
        else:
            self._config = SentimatrixConfig()

        # Apply overrides
        if llm_config:
            self._config = self._config.model_copy(update={"llm": llm_config})
        if scraper_config:
            self._config = self._config.model_copy(update={"scraper": scraper_config})
        if model_config:
            self._config = self._config.model_copy(update={"models": model_config})

        # Component instances (lazy initialized)
        self._sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self._emotion_detector: Optional[EmotionDetector] = None
        self._llm_manager = None  # Will be LLMProviderManager when initialized
        self._amazon_scraper = None
        self._steam_scraper = None
        self._youtube_scraper = None
        self._reddit_scraper = None
        self._httpx_scraper = None
        self._playwright_scraper = None

        # State
        self._initialized = False

        logger.debug("Sentimatrix instance created", config=str(self._config))

    @property
    def config(self) -> SentimatrixConfig:
        """Get current configuration."""
        return self._config

    @property
    def is_initialized(self) -> bool:
        """Check if Sentimatrix is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """
        Initialize Sentimatrix and all components.

        This prepares sentiment analysis and emotion detection models.
        Scrapers and LLM providers are lazily initialized when first used.

        Raises:
            ProviderInitializationError: If initialization fails
        """
        if self._initialized:
            return

        logger.info("Initializing Sentimatrix")

        try:
            # Initialize sentiment analyzer
            self._sentiment_analyzer = SentimentAnalyzer(
                config=self._config.models,
            )
            await self._sentiment_analyzer.initialize()

            # Initialize emotion detector
            self._emotion_detector = EmotionDetector(
                config=self._config.models,
            )
            await self._emotion_detector.initialize()

            self._initialized = True
            logger.info("Sentimatrix initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Sentimatrix: {e}")
            raise ProviderInitializationError("sentimatrix", str(e)) from e

    async def close(self) -> None:
        """
        Close Sentimatrix and cleanup all resources.
        """
        logger.info("Closing Sentimatrix")

        # Close sentiment analyzer
        if self._sentiment_analyzer:
            await self._sentiment_analyzer.close()
            self._sentiment_analyzer = None

        # Close emotion detector
        if self._emotion_detector:
            await self._emotion_detector.close()
            self._emotion_detector = None

        # Close LLM manager
        if self._llm_manager:
            await self._llm_manager.close()
            self._llm_manager = None

        # Close platform scrapers
        if self._amazon_scraper:
            await self._amazon_scraper.close()
            self._amazon_scraper = None

        if self._steam_scraper:
            await self._steam_scraper.close()
            self._steam_scraper = None

        if self._youtube_scraper:
            await self._youtube_scraper.close()
            self._youtube_scraper = None

        if self._reddit_scraper:
            await self._reddit_scraper.close()
            self._reddit_scraper = None

        # Close generic scrapers
        if self._httpx_scraper:
            await self._httpx_scraper.close()
            self._httpx_scraper = None

        if self._playwright_scraper:
            await self._playwright_scraper.close()
            self._playwright_scraper = None

        self._initialized = False
        logger.info("Sentimatrix closed")

    async def __aenter__(self) -> "Sentimatrix":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure Sentimatrix is initialized."""
        if not self._initialized:
            raise SentimatrixError(
                "Sentimatrix not initialized. Call initialize() or use async context manager."
            )

    # -------------------------------------------------------------------------
    # Sentiment Analysis Methods
    # -------------------------------------------------------------------------

    async def analyze_sentiment(
        self,
        text: str,
        return_all_scores: bool = True,
    ) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text to analyze
            return_all_scores: Include scores for all sentiment classes

        Returns:
            SentimentResult with sentiment label and confidence

        Example:
            >>> result = await sm.analyze_sentiment("I love this product!")
            >>> print(result.sentiment)  # "positive"
            >>> print(result.confidence)  # 0.95
        """
        self._ensure_initialized()
        return await self._sentiment_analyzer.analyze(text, return_all_scores)

    async def analyze_sentiment_batch(
        self,
        texts: List[str],
        return_all_scores: bool = True,
        batch_size: Optional[int] = None,
    ) -> BatchSentimentResult:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze
            return_all_scores: Include scores for all sentiment classes
            batch_size: Override default batch size

        Returns:
            BatchSentimentResult with aggregate statistics

        Example:
            >>> result = await sm.analyze_sentiment_batch([
            ...     "Great product!",
            ...     "Terrible service.",
            ...     "It's okay."
            ... ])
            >>> print(result.positive_count)  # 1
        """
        self._ensure_initialized()
        return await self._sentiment_analyzer.analyze_batch(
            texts, return_all_scores, batch_size
        )

    async def get_quick_sentiment(self, text: str) -> tuple[str, float]:
        """
        Quick sentiment analysis returning only label and score.

        Args:
            text: Input text

        Returns:
            Tuple of (sentiment_label, confidence_score)

        Example:
            >>> label, score = await sm.get_quick_sentiment("Great!")
            >>> print(label, score)  # "positive" 0.95
        """
        self._ensure_initialized()
        return await self._sentiment_analyzer.get_quick_sentiment(text)

    # -------------------------------------------------------------------------
    # Emotion Detection Methods
    # -------------------------------------------------------------------------

    async def detect_emotions(
        self,
        text: str,
        mode: Optional[EmotionMode] = None,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> EmotionResult:
        """
        Detect emotions in text.

        Args:
            text: Input text to analyze
            mode: Detection mode (single_label, multi_label, top_k)
            threshold: Score threshold for multi_label mode
            top_k: Number of emotions for top_k mode

        Returns:
            EmotionResult with detected emotions

        Example:
            >>> result = await sm.detect_emotions("I'm so happy!")
            >>> print(result.primary_emotion.label)  # "joy"
        """
        self._ensure_initialized()
        return await self._emotion_detector.detect(text, mode, threshold, top_k)

    async def detect_emotions_batch(
        self,
        texts: List[str],
        mode: Optional[EmotionMode] = None,
        threshold: Optional[float] = None,
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> BatchEmotionResult:
        """
        Detect emotions in multiple texts.

        Args:
            texts: List of texts to analyze
            mode: Detection mode
            threshold: Score threshold for multi_label mode
            top_k: Number of emotions for top_k mode
            batch_size: Override default batch size

        Returns:
            BatchEmotionResult with aggregate statistics
        """
        self._ensure_initialized()
        return await self._emotion_detector.detect_batch(
            texts, mode, threshold, top_k, batch_size
        )

    async def detect_ekman_emotions(self, text: str) -> Dict[str, float]:
        """
        Get Ekman's 6 basic emotions distribution.

        Args:
            text: Input text

        Returns:
            Dictionary mapping Ekman emotions to scores

        Example:
            >>> ekman = await sm.detect_ekman_emotions("I'm happy!")
            >>> print(ekman["joy"])  # 0.95
        """
        self._ensure_initialized()
        return await self._emotion_detector.detect_ekman(text)

    # -------------------------------------------------------------------------
    # Combined Analysis Methods
    # -------------------------------------------------------------------------

    async def analyze(
        self,
        text: str,
        include_emotions: bool = True,
    ) -> AnalysisResult:
        """
        Perform full analysis on text (sentiment + emotions).

        Args:
            text: Input text to analyze
            include_emotions: Also detect emotions

        Returns:
            AnalysisResult with sentiment and optional emotions

        Example:
            >>> result = await sm.analyze("I love this amazing product!")
            >>> print(result.sentiment.sentiment)  # "positive"
            >>> print(result.emotions.primary_emotion.label)  # "joy"
        """
        self._ensure_initialized()

        # Run sentiment analysis
        sentiment = await self._sentiment_analyzer.analyze(text)

        # Optionally run emotion detection
        emotions = None
        if include_emotions:
            emotions = await self._emotion_detector.detect(text)

        return AnalysisResult(
            text=text,
            sentiment=sentiment,
            emotions=emotions,
        )

    async def analyze_reviews(
        self,
        reviews: List[Review],
        include_emotions: bool = True,
    ) -> ReviewAnalysisResult:
        """
        Analyze a list of reviews.

        Args:
            reviews: List of Review objects to analyze
            include_emotions: Also detect emotions for each review

        Returns:
            ReviewAnalysisResult with aggregated statistics

        Example:
            >>> reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)
            >>> analysis = await sm.analyze_reviews(reviews)
            >>> print(f"Positive: {analysis.positive_ratio:.1%}")
        """
        self._ensure_initialized()

        if not reviews:
            return ReviewAnalysisResult(reviews=[])

        # Extract texts for batch processing
        texts = [r.text for r in reviews]

        # Batch sentiment analysis
        sentiment_batch = await self._sentiment_analyzer.analyze_batch(texts)

        # Batch emotion detection (if requested)
        emotion_batch = None
        if include_emotions:
            emotion_batch = await self._emotion_detector.detect_batch(texts)

        # Build results
        results: List[AnalysisResult] = []
        for i, review in enumerate(reviews):
            sentiment = sentiment_batch.results[i] if i < len(sentiment_batch.results) else None
            emotions = None
            if emotion_batch and i < len(emotion_batch.results):
                emotions = emotion_batch.results[i]

            results.append(AnalysisResult(
                text=review.text,
                sentiment=sentiment,
                emotions=emotions,
                metadata={
                    "review_id": review.id,
                    "author": review.author,
                    "rating": review.rating,
                    "timestamp": review.timestamp.isoformat() if review.timestamp else None,
                }
            ))

        return ReviewAnalysisResult(
            reviews=results,
            sentiment_summary=sentiment_batch.get_summary(),
            emotion_summary=emotion_batch.get_summary() if emotion_batch else None,
        )

    # -------------------------------------------------------------------------
    # Platform Scraping Methods
    # -------------------------------------------------------------------------

    async def _get_amazon_scraper(self):
        """Get or create Amazon scraper."""
        if self._amazon_scraper is None:
            from sentimatrix.providers.scrapers.platforms import AmazonScraper
            self._amazon_scraper = AmazonScraper()
            await self._amazon_scraper.initialize()
        return self._amazon_scraper

    async def _get_steam_scraper(self):
        """Get or create Steam scraper."""
        if self._steam_scraper is None:
            from sentimatrix.providers.scrapers.platforms import SteamScraper
            self._steam_scraper = SteamScraper()
            await self._steam_scraper.initialize()
        return self._steam_scraper

    async def _get_youtube_scraper(self, api_key: Optional[str] = None):
        """Get or create YouTube scraper."""
        if self._youtube_scraper is None:
            from sentimatrix.providers.scrapers.platforms.youtube import (
                YouTubeScraper,
                YouTubeConfig,
            )
            config = YouTubeConfig(api_key=api_key)
            self._youtube_scraper = YouTubeScraper(config)
            await self._youtube_scraper.initialize()
        return self._youtube_scraper

    async def _get_reddit_scraper(self):
        """Get or create Reddit scraper."""
        if self._reddit_scraper is None:
            from sentimatrix.providers.scrapers.platforms import RedditScraper
            self._reddit_scraper = RedditScraper()
            await self._reddit_scraper.initialize()
        return self._reddit_scraper

    async def scrape_amazon(
        self,
        asin: str,
        limit: int = 100,
        country: str = "us",
    ) -> List[Review]:
        """
        Scrape reviews from Amazon.

        Args:
            asin: Amazon Standard Identification Number
            limit: Maximum number of reviews to scrape
            country: Amazon country code (us, uk, de, etc.)

        Returns:
            List of Review objects

        Example:
            >>> reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)
            >>> print(f"Got {len(reviews)} reviews")
        """
        scraper = await self._get_amazon_scraper()
        return await scraper.scrape_reviews(asin, limit=limit, country=country)

    async def scrape_steam(
        self,
        app_id: str,
        limit: int = 100,
        language: str = "english",
    ) -> List[Review]:
        """
        Scrape reviews from Steam.

        Args:
            app_id: Steam application ID
            limit: Maximum number of reviews to scrape
            language: Language filter

        Returns:
            List of Review objects

        Example:
            >>> reviews = await sm.scrape_steam("730", limit=50)  # CS:GO
            >>> print(f"Got {len(reviews)} reviews")
        """
        scraper = await self._get_steam_scraper()
        return await scraper.scrape_reviews(app_id, limit=limit, language=language)

    async def scrape_youtube(
        self,
        video_id: str,
        limit: int = 100,
        api_key: Optional[str] = None,
    ) -> List[Review]:
        """
        Scrape comments from YouTube.

        Args:
            video_id: YouTube video ID
            limit: Maximum number of comments to scrape
            api_key: YouTube Data API key

        Returns:
            List of Review objects (comments)

        Example:
            >>> comments = await sm.scrape_youtube("dQw4w9WgXcQ", limit=50, api_key="...")
            >>> print(f"Got {len(comments)} comments")
        """
        scraper = await self._get_youtube_scraper(api_key)
        return await scraper.scrape_reviews(video_id, limit=limit)

    async def scrape_reddit(
        self,
        post_id: str,
        limit: int = 100,
    ) -> List[Review]:
        """
        Scrape comments from a Reddit post.

        Args:
            post_id: Reddit post ID
            limit: Maximum number of comments to scrape

        Returns:
            List of Review objects (comments)

        Example:
            >>> comments = await sm.scrape_reddit("abc123", limit=50)
            >>> print(f"Got {len(comments)} comments")
        """
        scraper = await self._get_reddit_scraper()
        return await scraper.scrape_reviews(post_id, limit=limit)

    # -------------------------------------------------------------------------
    # LLM Integration Methods
    # -------------------------------------------------------------------------

    async def _get_llm_manager(self):
        """Get or create LLM provider manager."""
        if self._llm_manager is None:
            from sentimatrix.providers.llm.manager import LLMProviderManager
            self._llm_manager = LLMProviderManager()

            # Add configured provider
            if self._config.llm.api_key:
                self._llm_manager.add_provider(
                    name=str(self._config.llm.provider),
                    config=self._config.llm,
                    priority=0,
                )

            await self._llm_manager.initialize()

        return self._llm_manager

    async def summarize_reviews(
        self,
        reviews: List[Review],
        max_reviews: int = 50,
        style: str = "concise",
    ) -> str:
        """
        Generate a summary of reviews using LLM.

        Args:
            reviews: List of reviews to summarize
            max_reviews: Maximum reviews to include in prompt
            style: Summary style ("concise", "detailed", "bullet_points")

        Returns:
            Summary text

        Raises:
            ConfigurationError: If no LLM provider is configured

        Example:
            >>> reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)
            >>> summary = await sm.summarize_reviews(reviews)
            >>> print(summary)
        """
        if not self._config.llm.api_key:
            raise ConfigurationError(
                "LLM provider not configured. Set llm.api_key in config."
            )

        manager = await self._get_llm_manager()

        # Prepare reviews for prompt
        review_texts = []
        for i, review in enumerate(reviews[:max_reviews]):
            rating_str = f"[{review.rating}/5]" if review.rating else ""
            review_texts.append(f"{i+1}. {rating_str} {review.text[:500]}")

        reviews_content = "\n".join(review_texts)

        # Build prompt based on style
        if style == "detailed":
            instruction = "Provide a detailed summary covering all major points."
        elif style == "bullet_points":
            instruction = "Summarize in bullet points with key takeaways."
        else:
            instruction = "Provide a concise 2-3 sentence summary."

        prompt = f"""Analyze the following customer reviews and {instruction}

Reviews:
{reviews_content}

Summary:"""

        response = await manager.generate(
            prompt=prompt,
            system_prompt="You are an expert at analyzing customer reviews and extracting key insights.",
            max_tokens=500,
        )

        return response.content.strip()

    async def generate_insights(
        self,
        reviews: List[Review],
        max_reviews: int = 50,
        analysis: Optional[ReviewAnalysisResult] = None,
    ) -> InsightsResult:
        """
        Generate comprehensive insights from reviews using LLM.

        Args:
            reviews: List of reviews to analyze
            max_reviews: Maximum reviews to include
            analysis: Optional pre-computed analysis results

        Returns:
            InsightsResult with summary, pros, cons, recommendations

        Raises:
            ConfigurationError: If no LLM provider is configured

        Example:
            >>> reviews = await sm.scrape_amazon("B08N5WRWNW", limit=50)
            >>> insights = await sm.generate_insights(reviews)
            >>> print("Pros:", insights.pros)
            >>> print("Cons:", insights.cons)
        """
        if not self._config.llm.api_key:
            raise ConfigurationError(
                "LLM provider not configured. Set llm.api_key in config."
            )

        manager = await self._get_llm_manager()

        # Run analysis if not provided
        if analysis is None:
            analysis = await self.analyze_reviews(reviews[:max_reviews])

        # Prepare reviews for prompt
        review_texts = []
        for i, review in enumerate(reviews[:max_reviews]):
            rating_str = f"[{review.rating}/5]" if review.rating else ""
            review_texts.append(f"{i+1}. {rating_str} {review.text[:400]}")

        reviews_content = "\n".join(review_texts)

        # Add sentiment context
        sentiment_context = f"""
Sentiment Analysis Summary:
- Total Reviews: {analysis.total_count}
- Positive: {analysis.positive_ratio:.1%}
- Negative: {analysis.negative_ratio:.1%}
- Average Polarity: {analysis.average_polarity:.2f}
"""

        prompt = f"""Analyze the following customer reviews and provide structured insights.

{sentiment_context}

Reviews:
{reviews_content}

Provide your analysis in the following format:

SUMMARY: (2-3 sentence overall summary)

KEY POINTS:
- Point 1
- Point 2
- Point 3

PROS:
- Pro 1
- Pro 2
- Pro 3

CONS:
- Con 1
- Con 2
- Con 3

RECOMMENDATIONS:
- Recommendation 1
- Recommendation 2

THEMES:
- Theme 1
- Theme 2"""

        response = await manager.generate(
            prompt=prompt,
            system_prompt="You are an expert analyst who extracts actionable insights from customer feedback.",
            max_tokens=1000,
        )

        # Parse response
        result = InsightsResult(raw_response=response.content)
        content = response.content

        # Extract sections
        sections = {
            "SUMMARY:": "summary",
            "KEY POINTS:": "key_points",
            "PROS:": "pros",
            "CONS:": "cons",
            "RECOMMENDATIONS:": "recommendations",
            "THEMES:": "themes",
        }

        current_section = None
        current_content: List[str] = []

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check for section header
            section_found = False
            for header, field in sections.items():
                if line.upper().startswith(header.upper()):
                    # Save previous section
                    if current_section:
                        self._save_section(result, current_section, current_content)
                    current_section = field
                    current_content = []
                    # Check for inline content
                    remainder = line[len(header):].strip()
                    if remainder:
                        current_content.append(remainder)
                    section_found = True
                    break

            if not section_found and current_section:
                # Remove bullet points
                if line.startswith("-") or line.startswith("•"):
                    line = line[1:].strip()
                if line:
                    current_content.append(line)

        # Save last section
        if current_section:
            self._save_section(result, current_section, current_content)

        return result

    def _save_section(
        self,
        result: InsightsResult,
        section: str,
        content: List[str],
    ) -> None:
        """Save parsed section to InsightsResult."""
        if section == "summary":
            result.summary = " ".join(content)
        elif section == "key_points":
            result.key_points = content
        elif section == "pros":
            result.pros = content
        elif section == "cons":
            result.cons = content
        elif section == "recommendations":
            result.recommendations = content
        elif section == "themes":
            result.themes = content

    async def compare_products(
        self,
        item_a_reviews: List[Review],
        item_b_reviews: List[Review],
        item_a_name: str = "Product A",
        item_b_name: str = "Product B",
        max_reviews_per_item: int = 30,
    ) -> ComparisonResult:
        """
        Compare two products/items based on their reviews.

        Args:
            item_a_reviews: Reviews for first item
            item_b_reviews: Reviews for second item
            item_a_name: Name/identifier for first item
            item_b_name: Name/identifier for second item
            max_reviews_per_item: Maximum reviews per item

        Returns:
            ComparisonResult with analysis and comparison

        Example:
            >>> reviews_a = await sm.scrape_amazon("ASIN_A", limit=30)
            >>> reviews_b = await sm.scrape_amazon("ASIN_B", limit=30)
            >>> comparison = await sm.compare_products(
            ...     reviews_a, reviews_b,
            ...     "iPhone 15", "Samsung S24"
            ... )
            >>> print(f"Winner: {comparison.winner}")
        """
        # Analyze both items
        analysis_a = await self.analyze_reviews(item_a_reviews[:max_reviews_per_item])
        analysis_b = await self.analyze_reviews(item_b_reviews[:max_reviews_per_item])

        result = ComparisonResult(
            item_a=item_a_name,
            item_b=item_b_name,
            item_a_analysis=analysis_a,
            item_b_analysis=analysis_b,
        )

        # Determine winner based on polarity
        if analysis_a.average_polarity > analysis_b.average_polarity + 0.1:
            result.winner = item_a_name
        elif analysis_b.average_polarity > analysis_a.average_polarity + 0.1:
            result.winner = item_b_name
        else:
            result.winner = "tie"

        # Generate comparison with LLM if configured
        if self._config.llm.api_key:
            try:
                manager = await self._get_llm_manager()

                # Prepare brief review samples
                a_samples = [r.text[:200] for r in item_a_reviews[:10]]
                b_samples = [r.text[:200] for r in item_b_reviews[:10]]

                prompt = f"""Compare these two products based on customer reviews:

{item_a_name} (Positive: {analysis_a.positive_ratio:.1%}, Avg Score: {analysis_a.average_polarity:.2f}):
Sample reviews: {a_samples[:3]}

{item_b_name} (Positive: {analysis_b.positive_ratio:.1%}, Avg Score: {analysis_b.average_polarity:.2f}):
Sample reviews: {b_samples[:3]}

Provide a brief comparison highlighting key differences and which product customers prefer overall."""

                response = await manager.generate(
                    prompt=prompt,
                    system_prompt="You are a product comparison expert.",
                    max_tokens=300,
                )
                result.comparison_summary = response.content.strip()

            except Exception as e:
                logger.warning(f"Failed to generate comparison summary: {e}")
                result.comparison_summary = (
                    f"Based on sentiment analysis, {result.winner} has more positive reviews."
                )
        else:
            result.comparison_summary = (
                f"Based on sentiment analysis, {item_a_name} has {analysis_a.positive_ratio:.1%} positive reviews "
                f"while {item_b_name} has {analysis_b.positive_ratio:.1%} positive reviews."
            )

        return result

    # -------------------------------------------------------------------------
    # Pipeline Methods
    # -------------------------------------------------------------------------

    async def run_analysis_pipeline(
        self,
        url_or_id: str,
        platform: str = "auto",
        limit: int = 100,
        include_insights: bool = True,
    ) -> Dict[str, Any]:
        """
        Run a complete analysis pipeline for a URL or product ID.

        This method:
        1. Detects the platform (or uses specified)
        2. Scrapes reviews
        3. Analyzes sentiment and emotions
        4. Generates insights (if LLM configured)

        Args:
            url_or_id: URL or product identifier
            platform: Platform name or "auto" to detect
            limit: Maximum reviews to scrape
            include_insights: Generate LLM insights

        Returns:
            Dictionary with all analysis results

        Example:
            >>> result = await sm.run_analysis_pipeline(
            ...     "https://amazon.com/dp/B08N5WRWNW",
            ...     limit=50
            ... )
            >>> print(result["summary"])
        """
        self._ensure_initialized()

        # Detect platform if auto
        if platform == "auto":
            platform = self._detect_platform(url_or_id)

        # Create pipeline
        pipeline = Pipeline(
            "analysis_pipeline",
            description=f"Full analysis for {platform}",
        )

        # Step 1: Scrape reviews
        async def scrape_step(_, context: PipelineContext) -> List[Review]:
            identifier = context.get("identifier")
            plat = context.get("platform")
            lim = context.get("limit", 100)

            if plat == "amazon":
                return await self.scrape_amazon(identifier, limit=lim)
            elif plat == "steam":
                return await self.scrape_steam(identifier, limit=lim)
            elif plat == "youtube":
                return await self.scrape_youtube(identifier, limit=lim)
            elif plat == "reddit":
                return await self.scrape_reddit(identifier, limit=lim)
            else:
                raise ValidationError(f"Unknown platform: {plat}")

        pipeline.add_step(FunctionStep("scrape", scrape_step))

        # Step 2: Analyze reviews
        async def analyze_step(reviews: List[Review], context: PipelineContext) -> ReviewAnalysisResult:
            context.set("reviews", reviews)
            return await self.analyze_reviews(reviews)

        pipeline.add_step(FunctionStep("analyze", analyze_step))

        # Step 3: Generate insights (optional)
        if include_insights and self._config.llm.api_key:
            async def insights_step(analysis: ReviewAnalysisResult, context: PipelineContext) -> Dict[str, Any]:
                reviews = context.get("reviews", [])
                insights = await self.generate_insights(reviews, analysis=analysis)
                return {
                    "analysis": analysis,
                    "insights": insights,
                }

            pipeline.add_step(FunctionStep("insights", insights_step))

        # Run pipeline
        context = PipelineContext({
            "identifier": self._extract_identifier(url_or_id, platform),
            "platform": platform,
            "limit": limit,
        })

        result = await pipeline.run(context=context)

        if not result.success:
            raise SentimatrixError(f"Pipeline failed: {result.error}")

        # Format output
        output = result.output
        if isinstance(output, dict):
            return {
                "platform": platform,
                "identifier": context.get("identifier"),
                "reviews_count": len(context.get("reviews", [])),
                "analysis": output.get("analysis", output).to_dict() if hasattr(output.get("analysis", output), "to_dict") else output,
                "insights": output.get("insights", InsightsResult()).to_dict() if "insights" in output else None,
                "pipeline_duration_ms": result.total_duration_ms,
            }
        elif isinstance(output, ReviewAnalysisResult):
            return {
                "platform": platform,
                "identifier": context.get("identifier"),
                "reviews_count": len(context.get("reviews", [])),
                "analysis": output.to_dict(),
                "insights": None,
                "pipeline_duration_ms": result.total_duration_ms,
            }

        return {"output": output, "pipeline_duration_ms": result.total_duration_ms}

    def _detect_platform(self, url_or_id: str) -> str:
        """Detect platform from URL or ID."""
        url_lower = url_or_id.lower()

        if "amazon" in url_lower:
            return "amazon"
        elif "steam" in url_lower or "steampowered" in url_lower:
            return "steam"
        elif "youtube" in url_lower or "youtu.be" in url_lower:
            return "youtube"
        elif "reddit" in url_lower or "redd.it" in url_lower:
            return "reddit"

        # Check if it looks like an ASIN (10 alphanumeric chars)
        if len(url_or_id) == 10 and url_or_id.isalnum():
            return "amazon"

        # Check if it looks like a Steam app ID
        if url_or_id.isdigit():
            return "steam"

        # Check if it looks like a YouTube video ID (11 chars)
        if len(url_or_id) == 11:
            return "youtube"

        raise ValidationError(
            f"Cannot detect platform from '{url_or_id}'. "
            "Please specify platform explicitly."
        )

    def _extract_identifier(self, url_or_id: str, platform: str) -> str:
        """Extract product/content identifier from URL or return as-is."""
        import re

        if platform == "amazon":
            # Extract ASIN from URL
            match = re.search(r"/(?:dp|gp/product)/([A-Z0-9]{10})", url_or_id, re.I)
            if match:
                return match.group(1)
            return url_or_id

        elif platform == "steam":
            # Extract app ID from URL
            match = re.search(r"/app/(\d+)", url_or_id)
            if match:
                return match.group(1)
            return url_or_id

        elif platform == "youtube":
            # Extract video ID from URL
            patterns = [
                r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})",
                r"/embed/([a-zA-Z0-9_-]{11})",
                r"/shorts/([a-zA-Z0-9_-]{11})",
            ]
            for pattern in patterns:
                match = re.search(pattern, url_or_id)
                if match:
                    return match.group(1)
            return url_or_id

        elif platform == "reddit":
            # Extract post ID from URL
            match = re.search(r"/comments/([a-z0-9]+)", url_or_id, re.I)
            if match:
                return match.group(1)
            return url_or_id

        return url_or_id

    # -------------------------------------------------------------------------
    # Export Methods
    # -------------------------------------------------------------------------

    async def export_to_json(
        self,
        data: Any,
        path: str,
        pretty_print: bool = True,
    ) -> "ExportResult":
        """
        Export analysis results to JSON file.

        Args:
            data: Data to export (analysis result, reviews, etc.)
            path: Output file path
            pretty_print: Pretty print JSON output

        Returns:
            ExportResult with operation details

        Example:
            >>> result = await sm.analyze_reviews(reviews)
            >>> export = await sm.export_to_json(result, "results.json")
            >>> print(f"Exported to {export.path}")
        """
        from sentimatrix.output.exporters import export_to_json as _export_json

        return await _export_json(data, path, pretty_print=pretty_print)

    async def export_to_csv(
        self,
        data: Any,
        path: str,
        columns: Optional[List[str]] = None,
    ) -> "ExportResult":
        """
        Export analysis results to CSV file.

        Args:
            data: Data to export (list of reviews/results)
            path: Output file path
            columns: Column names to include (None = auto-detect)

        Returns:
            ExportResult with operation details

        Example:
            >>> result = await sm.analyze_reviews(reviews)
            >>> export = await sm.export_to_csv(result.reviews, "results.csv")
        """
        from sentimatrix.output.exporters import export_to_csv as _export_csv

        return await _export_csv(data, path, columns=columns)

    async def export_to_excel(
        self,
        data: Any,
        path: str,
        sheet_name: str = "Analysis",
    ) -> "ExportResult":
        """
        Export analysis results to Excel file.

        Args:
            data: Data to export
            path: Output file path (.xlsx)
            sheet_name: Excel sheet name

        Returns:
            ExportResult with operation details

        Example:
            >>> result = await sm.analyze_reviews(reviews)
            >>> export = await sm.export_to_excel(result, "report.xlsx")
        """
        from sentimatrix.output.exporters import export_to_excel as _export_excel

        return await _export_excel(data, path, sheet_name=sheet_name)

    async def generate_html_report(
        self,
        data: Any,
        path: Optional[str] = None,
        title: str = "Sentimatrix Analysis Report",
        theme: str = "default",
    ) -> str:
        """
        Generate an HTML report from analysis results.

        Args:
            data: Data to format (analysis result, reviews, etc.)
            path: Optional output file path (if None, returns HTML string)
            title: Report title
            theme: Visual theme ("default", "dark", "light")

        Returns:
            HTML string

        Example:
            >>> result = await sm.analyze_reviews(reviews)
            >>> html = await sm.generate_html_report(result, "report.html")
        """
        from sentimatrix.output.formatters import HTMLFormatter, FormatOptions

        options = FormatOptions(title=title, theme=theme)
        formatter = HTMLFormatter(options)
        html_content = await formatter.format(data)

        if path:
            await formatter.save(html_content, path)

        return html_content

    async def create_sentiment_chart(
        self,
        data: Any,
        path: Optional[str] = None,
        chart_type: str = "bar",
        title: Optional[str] = None,
    ) -> Any:
        """
        Create a sentiment visualization chart.

        Args:
            data: Analysis result with sentiment data
            path: Optional output file path (if None, returns figure)
            chart_type: Chart type ("bar", "pie", "donut")
            title: Chart title

        Returns:
            Matplotlib figure (or VisualizationResult if path provided)

        Example:
            >>> result = await sm.analyze_reviews(reviews)
            >>> await sm.create_sentiment_chart(result, "chart.png")
        """
        from sentimatrix.output.visualizers import ChartVisualizer, VisualizationOptions

        options = VisualizationOptions(title=title)
        visualizer = ChartVisualizer(options)

        if chart_type == "bar":
            fig = await visualizer.create_sentiment_bar_chart(data, title=title)
        elif chart_type == "pie":
            fig = await visualizer.create_sentiment_pie_chart(data, title=title, donut=False)
        elif chart_type == "donut":
            fig = await visualizer.create_sentiment_pie_chart(data, title=title, donut=True)
        else:
            raise ValueError(f"Unknown chart type: {chart_type}")

        if path:
            return await visualizer.save(fig, path)

        return fig

    async def create_emotion_chart(
        self,
        data: Any,
        path: Optional[str] = None,
        title: Optional[str] = None,
        top_k: int = 8,
    ) -> Any:
        """
        Create an emotion visualization chart.

        Args:
            data: Analysis result with emotion data
            path: Optional output file path
            title: Chart title
            top_k: Number of top emotions to show

        Returns:
            Matplotlib figure (or VisualizationResult if path provided)

        Example:
            >>> result = await sm.analyze_reviews(reviews)
            >>> await sm.create_emotion_chart(result, "emotions.png")
        """
        from sentimatrix.output.visualizers import ChartVisualizer, VisualizationOptions

        options = VisualizationOptions(title=title)
        visualizer = ChartVisualizer(options)
        fig = await visualizer.create_emotion_bar_chart(data, title=title, top_k=top_k)

        if path:
            return await visualizer.save(fig, path)

        return fig

    async def create_comparison_chart(
        self,
        data: "ComparisonResult",
        path: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Any:
        """
        Create a product comparison chart.

        Args:
            data: ComparisonResult from compare_products()
            path: Optional output file path
            title: Chart title

        Returns:
            Matplotlib figure (or VisualizationResult if path provided)

        Example:
            >>> comparison = await sm.compare_products(reviews_a, reviews_b)
            >>> await sm.create_comparison_chart(comparison, "comparison.png")
        """
        from sentimatrix.output.visualizers import ChartVisualizer, VisualizationOptions

        options = VisualizationOptions(title=title)
        visualizer = ChartVisualizer(options)
        fig = await visualizer.create_comparison_chart(data, title=title)

        if path:
            return await visualizer.save(fig, path)

        return fig

    # -------------------------------------------------------------------------
    # Multi-Modal Methods
    # -------------------------------------------------------------------------

    async def analyze_audio(
        self,
        audio_input: Union[str, bytes],
        language: Optional[str] = None,
        include_emotions: bool = True,
        engine: str = "whisper",
    ) -> "AudioAnalysisResult":
        """
        Analyze sentiment from audio input.

        Pipeline:
        1. Transcribe audio to text
        2. Analyze text sentiment
        3. Optionally detect emotions

        Args:
            audio_input: Audio file path or bytes
            language: Language code (auto-detect if None)
            include_emotions: Also detect emotions
            engine: Transcription engine (whisper, groq_whisper, whisper_api)

        Returns:
            AudioAnalysisResult with transcription and sentiment

        Example:
            >>> result = await sm.analyze_audio("interview.mp3")
            >>> print(f"Transcription: {result.transcription.text}")
            >>> print(f"Sentiment: {result.sentiment.sentiment}")
        """
        from sentimatrix.analysis.multimodal import MultiModalAnalyzer

        analyzer = MultiModalAnalyzer(
            audio_engine=engine,
            openai_api_key=self._config.llm.api_key if self._config.llm.provider == "openai" else None,
            groq_api_key=self._config.llm.api_key if self._config.llm.provider == "groq" else None,
        )
        await analyzer.initialize()

        try:
            return await analyzer.analyze_audio(
                audio_input,
                language=language,
                include_emotions=include_emotions,
            )
        finally:
            await analyzer.close()

    async def analyze_image(
        self,
        image_input: Union[str, bytes],
        prompt: Optional[str] = None,
        detailed: bool = False,
        include_emotions: bool = True,
        model: str = "gpt4v",
    ) -> "ImageAnalysisResult":
        """
        Analyze sentiment from image input.

        Pipeline:
        1. Generate image caption
        2. Analyze caption sentiment
        3. Optionally detect emotions

        Args:
            image_input: Image file path or bytes
            prompt: Custom captioning prompt
            detailed: Generate detailed caption
            include_emotions: Also detect emotions
            model: Captioning model (gpt4v, claude_vision, gemini_vision)

        Returns:
            ImageAnalysisResult with caption and sentiment

        Example:
            >>> result = await sm.analyze_image("product.jpg")
            >>> print(f"Caption: {result.caption.caption}")
            >>> print(f"Sentiment: {result.sentiment.sentiment}")
        """
        from sentimatrix.analysis.multimodal import MultiModalAnalyzer

        analyzer = MultiModalAnalyzer(
            image_model=model,
            openai_api_key=self._config.llm.api_key if self._config.llm.provider == "openai" else None,
            anthropic_api_key=self._config.llm.api_key if self._config.llm.provider == "anthropic" else None,
            google_api_key=self._config.llm.api_key if self._config.llm.provider == "gemini" else None,
        )
        await analyzer.initialize()

        try:
            return await analyzer.analyze_image(
                image_input,
                prompt=prompt,
                detailed=detailed,
                include_emotions=include_emotions,
            )
        finally:
            await analyzer.close()

    async def analyze_video(
        self,
        video_path: str,
        analyze_audio: bool = True,
        analyze_frames: bool = True,
        max_frames: int = 10,
    ) -> "VideoAnalysisResult":
        """
        Analyze sentiment from video input.

        Pipeline:
        1. Extract key frames
        2. Optionally extract and analyze audio
        3. Analyze frame captions for sentiment
        4. Combine results

        Args:
            video_path: Path to video file
            analyze_audio: Also analyze audio track
            analyze_frames: Analyze extracted frames
            max_frames: Maximum frames to analyze

        Returns:
            VideoAnalysisResult with combined analysis

        Example:
            >>> result = await sm.analyze_video("review.mp4")
            >>> print(f"Combined sentiment: {result.combined_sentiment.sentiment}")
            >>> print(f"Frames analyzed: {result.frames_analyzed}")
        """
        from sentimatrix.analysis.multimodal import MultiModalAnalyzer

        analyzer = MultiModalAnalyzer(
            openai_api_key=self._config.llm.api_key if self._config.llm.provider == "openai" else None,
            groq_api_key=self._config.llm.api_key if self._config.llm.provider == "groq" else None,
        )
        await analyzer.initialize()

        try:
            return await analyzer.analyze_video(
                video_path,
                analyze_audio=analyze_audio,
                analyze_frames=analyze_frames,
                max_frames=max_frames,
            )
        finally:
            await analyzer.close()

    async def transcribe_audio(
        self,
        audio_input: Union[str, bytes],
        language: Optional[str] = None,
        include_timestamps: bool = False,
        engine: str = "whisper",
    ) -> "TranscriptionResult":
        """
        Transcribe audio to text without sentiment analysis.

        Args:
            audio_input: Audio file path or bytes
            language: Language code (auto-detect if None)
            include_timestamps: Include segment timestamps
            engine: Transcription engine

        Returns:
            TranscriptionResult with transcribed text

        Example:
            >>> result = await sm.transcribe_audio("speech.mp3")
            >>> print(result.text)
        """
        from sentimatrix.input.handlers import AudioHandler

        handler = AudioHandler(engine=engine)
        await handler.initialize()

        try:
            return await handler.transcribe(
                audio_input,
                language=language,
                include_timestamps=include_timestamps,
            )
        finally:
            await handler.close()

    async def caption_image(
        self,
        image_input: Union[str, bytes],
        prompt: Optional[str] = None,
        detailed: bool = False,
        model: str = "gpt4v",
    ) -> "CaptionResult":
        """
        Generate caption for an image without sentiment analysis.

        Args:
            image_input: Image file path or bytes
            prompt: Custom prompt for captioning
            detailed: Generate detailed description
            model: Captioning model

        Returns:
            CaptionResult with generated caption

        Example:
            >>> result = await sm.caption_image("product.jpg")
            >>> print(result.caption)
        """
        from sentimatrix.input.handlers import ImageHandler

        handler = ImageHandler(
            model=model,
            api_key=self._config.llm.api_key if self._config.llm.provider in ("openai", "anthropic", "gemini") else None,
        )
        await handler.initialize()

        try:
            return await handler.caption(
                image_input,
                prompt=prompt,
                detailed=detailed,
            )
        finally:
            await handler.close()

    async def analyze_multimodal(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[str, bytes]] = None,
        image: Optional[Union[str, bytes]] = None,
    ) -> "MultiModalResult":
        """
        Analyze multiple modalities and combine results.

        Args:
            text: Text input
            audio: Audio file path or bytes
            image: Image file path or bytes

        Returns:
            MultiModalResult with combined analysis

        Example:
            >>> result = await sm.analyze_multimodal(
            ...     text="Great product!",
            ...     image="product.jpg",
            ... )
            >>> print(f"Combined sentiment: {result.combined_sentiment.sentiment}")
        """
        from sentimatrix.analysis.multimodal import MultiModalAnalyzer

        analyzer = MultiModalAnalyzer(
            openai_api_key=self._config.llm.api_key if self._config.llm.provider == "openai" else None,
            anthropic_api_key=self._config.llm.api_key if self._config.llm.provider == "anthropic" else None,
            google_api_key=self._config.llm.api_key if self._config.llm.provider == "gemini" else None,
            groq_api_key=self._config.llm.api_key if self._config.llm.provider == "groq" else None,
        )
        await analyzer.initialize()

        try:
            return await analyzer.analyze_multimodal(
                text=text,
                audio=audio,
                image=image,
            )
        finally:
            await analyzer.close()


# Convenience function
def create_sentimatrix(
    config: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Sentimatrix:
    """
    Create a Sentimatrix instance.

    Args:
        config: Configuration dictionary
        **kwargs: Additional configuration options

    Returns:
        Sentimatrix instance (not initialized)

    Example:
        >>> sm = create_sentimatrix({"llm": {"api_key": "sk-..."}})
        >>> async with sm:
        ...     result = await sm.analyze_sentiment("Great!")
    """
    full_config = config or {}
    full_config.update(kwargs)
    return Sentimatrix(config=full_config)


__all__ = [
    "Sentimatrix",
    "AnalysisResult",
    "ReviewAnalysisResult",
    "InsightsResult",
    "ComparisonResult",
    "create_sentimatrix",
]
