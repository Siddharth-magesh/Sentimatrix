"""
Sentimatrix Sentiment Analysis Module

Provides comprehensive sentiment analysis functionality including:
- Quick sentiment analysis (positive/negative/neutral)
- Fine-grained sentiment (5-class)
- Batch processing
- Structured sentiment results with confidence scores

Example:
    >>> analyzer = SentimentAnalyzer()
    >>> await analyzer.initialize()
    >>> result = await analyzer.analyze("I love this product!")
    >>> print(result.sentiment, result.confidence)
    positive 0.95
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from sentimatrix.core.config import ModelConfig, SentimatrixConfig
from sentimatrix.core.exceptions import (
    InvalidInputError,
    ModelInferenceError,
    ModelLoadError,
    SentimatrixError,
    ValidationError,
)
from sentimatrix.providers.base import PredictionResult
from sentimatrix.providers.models.huggingface import (
    HuggingFaceModelProvider,
    ModelType,
    SentimentModelProvider,
)


class SentimentLabel(str, Enum):
    """Standardized sentiment labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    VERY_POSITIVE = "very_positive"
    VERY_NEGATIVE = "very_negative"


class SentimentClass(str, Enum):
    """Sentiment classification modes."""

    THREE_CLASS = "three_class"  # positive, neutral, negative
    FIVE_CLASS = "five_class"  # very_negative, negative, neutral, positive, very_positive
    BINARY = "binary"  # positive, negative


# Mapping from model labels to standardized labels
LABEL_MAPPING = {
    # CardiffNLP RoBERTa model
    "LABEL_0": SentimentLabel.NEGATIVE,
    "LABEL_1": SentimentLabel.NEUTRAL,
    "LABEL_2": SentimentLabel.POSITIVE,
    "negative": SentimentLabel.NEGATIVE,
    "neutral": SentimentLabel.NEUTRAL,
    "positive": SentimentLabel.POSITIVE,
    # NLPtown 5-star model
    "1 star": SentimentLabel.VERY_NEGATIVE,
    "2 stars": SentimentLabel.NEGATIVE,
    "3 stars": SentimentLabel.NEUTRAL,
    "4 stars": SentimentLabel.POSITIVE,
    "5 stars": SentimentLabel.VERY_POSITIVE,
}


@dataclass
class SentimentResult:
    """
    Result of sentiment analysis.

    Attributes:
        text: Original input text
        sentiment: Standardized sentiment label
        confidence: Confidence score (0-1)
        raw_label: Original model label
        raw_score: Original model score
        all_scores: Scores for all labels
        model_name: Model used for analysis
        processing_time_ms: Processing time in milliseconds
    """

    text: str
    sentiment: SentimentLabel
    confidence: float
    raw_label: str = ""
    raw_score: float = 0.0
    all_scores: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    processing_time_ms: float = 0.0

    @property
    def is_positive(self) -> bool:
        """Check if sentiment is positive."""
        return self.sentiment in (SentimentLabel.POSITIVE, SentimentLabel.VERY_POSITIVE)

    @property
    def is_negative(self) -> bool:
        """Check if sentiment is negative."""
        return self.sentiment in (SentimentLabel.NEGATIVE, SentimentLabel.VERY_NEGATIVE)

    @property
    def is_neutral(self) -> bool:
        """Check if sentiment is neutral."""
        return self.sentiment == SentimentLabel.NEUTRAL

    @property
    def polarity(self) -> float:
        """
        Get polarity score (-1 to 1).

        -1 = very negative
        0 = neutral
        1 = very positive
        """
        polarity_map = {
            SentimentLabel.VERY_NEGATIVE: -1.0,
            SentimentLabel.NEGATIVE: -0.5,
            SentimentLabel.NEUTRAL: 0.0,
            SentimentLabel.POSITIVE: 0.5,
            SentimentLabel.VERY_POSITIVE: 1.0,
        }
        return polarity_map.get(self.sentiment, 0.0) * self.confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "sentiment": self.sentiment.value,
            "confidence": self.confidence,
            "raw_label": self.raw_label,
            "raw_score": self.raw_score,
            "all_scores": self.all_scores,
            "model_name": self.model_name,
            "processing_time_ms": self.processing_time_ms,
            "is_positive": self.is_positive,
            "is_negative": self.is_negative,
            "polarity": self.polarity,
        }


@dataclass
class BatchSentimentResult:
    """
    Result of batch sentiment analysis.

    Attributes:
        results: List of individual sentiment results
        total_count: Total number of texts analyzed
        positive_count: Number of positive results
        negative_count: Number of negative results
        neutral_count: Number of neutral results
        average_confidence: Average confidence across all results
        total_processing_time_ms: Total processing time
    """

    results: List[SentimentResult]
    total_count: int = 0
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    average_confidence: float = 0.0
    total_processing_time_ms: float = 0.0

    def __post_init__(self) -> None:
        """Calculate aggregate statistics."""
        if not self.results:
            return

        self.total_count = len(self.results)
        self.positive_count = sum(1 for r in self.results if r.is_positive)
        self.negative_count = sum(1 for r in self.results if r.is_negative)
        self.neutral_count = sum(1 for r in self.results if r.is_neutral)
        self.average_confidence = (
            sum(r.confidence for r in self.results) / len(self.results)
        )
        self.total_processing_time_ms = sum(r.processing_time_ms for r in self.results)

    @property
    def positive_ratio(self) -> float:
        """Get ratio of positive results."""
        return self.positive_count / self.total_count if self.total_count else 0.0

    @property
    def negative_ratio(self) -> float:
        """Get ratio of negative results."""
        return self.negative_count / self.total_count if self.total_count else 0.0

    @property
    def neutral_ratio(self) -> float:
        """Get ratio of neutral results."""
        return self.neutral_count / self.total_count if self.total_count else 0.0

    @property
    def average_polarity(self) -> float:
        """Get average polarity across all results."""
        if not self.results:
            return 0.0
        return sum(r.polarity for r in self.results) / len(self.results)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "neutral_count": self.neutral_count,
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "neutral_ratio": self.neutral_ratio,
            "average_confidence": self.average_confidence,
            "average_polarity": self.average_polarity,
            "total_processing_time_ms": self.total_processing_time_ms,
            "results": [r.to_dict() for r in self.results],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary without individual results."""
        result = self.to_dict()
        del result["results"]
        return result


class SentimentAnalyzer:
    """
    Sentiment analysis engine.

    Provides methods for analyzing sentiment in text using HuggingFace
    models with support for:
    - Single text analysis
    - Batch processing
    - Multiple sentiment classification modes
    - Configurable models

    Example:
        >>> analyzer = SentimentAnalyzer()
        >>> await analyzer.initialize()
        >>> result = await analyzer.analyze("Great product!")
        >>> print(result.sentiment)
        positive
    """

    # Default models for different classification modes
    DEFAULT_MODELS = {
        SentimentClass.THREE_CLASS: "cardiffnlp/twitter-roberta-base-sentiment-latest",
        SentimentClass.FIVE_CLASS: "nlptown/bert-base-multilingual-uncased-sentiment",
        SentimentClass.BINARY: "distilbert-base-uncased-finetuned-sst-2-english",
    }

    def __init__(
        self,
        config: Optional[Union[ModelConfig, SentimatrixConfig]] = None,
        model_name: Optional[str] = None,
        classification_mode: SentimentClass = SentimentClass.THREE_CLASS,
    ) -> None:
        """
        Initialize sentiment analyzer.

        Args:
            config: Model or Sentimatrix configuration
            model_name: Override model name
            classification_mode: Sentiment classification mode
        """
        # Handle different config types
        if isinstance(config, SentimatrixConfig):
            self._config = config.models
        else:
            self._config = config or ModelConfig()

        self._classification_mode = classification_mode
        self._model_name = model_name or self._get_default_model()
        self._provider: Optional[SentimentModelProvider] = None
        self._initialized = False

    def _get_default_model(self) -> str:
        """Get default model for current classification mode."""
        return self.DEFAULT_MODELS.get(
            self._classification_mode,
            self._config.sentiment_model,
        )

    async def initialize(self) -> None:
        """
        Initialize the analyzer.

        Loads the model and prepares for analysis.

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        if self._initialized:
            return

        # Create and initialize provider
        config_with_model = ModelConfig(
            sentiment_model=self._model_name,
            device=self._config.device,
            batch_size=self._config.batch_size,
            max_length=self._config.max_length,
            use_quantization=self._config.use_quantization,
            cache_models=self._config.cache_models,
        )

        self._provider = SentimentModelProvider(config=config_with_model)
        await self._provider.initialize()
        self._initialized = True

    async def close(self) -> None:
        """Cleanup resources."""
        if self._provider:
            await self._provider.close()
        self._initialized = False

    async def __aenter__(self) -> "SentimentAnalyzer":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure analyzer is initialized."""
        if not self._initialized:
            raise SentimatrixError(
                "Analyzer not initialized. Call initialize() first."
            )

    def _normalize_label(self, raw_label: str) -> SentimentLabel:
        """
        Normalize raw model label to standardized label.

        Args:
            raw_label: Label from model

        Returns:
            Standardized sentiment label
        """
        # Try direct mapping
        if raw_label in LABEL_MAPPING:
            return LABEL_MAPPING[raw_label]

        # Try lowercase
        raw_lower = raw_label.lower()
        if raw_lower in LABEL_MAPPING:
            return LABEL_MAPPING[raw_lower]

        # Try to infer from label name
        if "positive" in raw_lower or "pos" in raw_lower:
            if "very" in raw_lower or "strong" in raw_lower:
                return SentimentLabel.VERY_POSITIVE
            return SentimentLabel.POSITIVE
        if "negative" in raw_lower or "neg" in raw_lower:
            if "very" in raw_lower or "strong" in raw_lower:
                return SentimentLabel.VERY_NEGATIVE
            return SentimentLabel.NEGATIVE
        if "neutral" in raw_lower or "neu" in raw_lower:
            return SentimentLabel.NEUTRAL

        # Default to neutral if unknown
        return SentimentLabel.NEUTRAL

    def _validate_text(self, text: str) -> str:
        """
        Validate and preprocess input text.

        Args:
            text: Input text

        Returns:
            Validated and cleaned text

        Raises:
            InvalidInputError: If text is invalid
        """
        if text is None:
            raise InvalidInputError("text", "Text cannot be None")

        if not isinstance(text, str):
            raise InvalidInputError("text", f"Text must be a string, got {type(text).__name__}")

        # Strip whitespace
        text = text.strip()

        # Check for empty text
        if not text:
            raise InvalidInputError("text", "Text cannot be empty")

        # Check max length
        if len(text) > 100000:
            raise InvalidInputError(
                "text",
                f"Text too long: {len(text)} characters (max 100,000)",
            )

        return text

    async def analyze(
        self,
        text: str,
        return_all_scores: bool = True,
    ) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Input text to analyze
            return_all_scores: Include scores for all labels

        Returns:
            SentimentResult with analysis results

        Raises:
            InvalidInputError: If text is invalid
            ModelInferenceError: If analysis fails

        Example:
            >>> result = await analyzer.analyze("I love this product!")
            >>> print(result.sentiment, result.confidence)
            positive 0.95
        """
        self._ensure_initialized()

        try:
            text = self._validate_text(text)
        except InvalidInputError:
            # Return neutral result for invalid input with warning
            return SentimentResult(
                text=text if isinstance(text, str) else "",
                sentiment=SentimentLabel.NEUTRAL,
                confidence=0.0,
                raw_label="",
                raw_score=0.0,
                all_scores={},
                model_name=self._model_name,
                processing_time_ms=0.0,
            )

        # Run prediction
        prediction = await self._provider.predict(
            text,
            return_all_scores=return_all_scores,
        )

        # Normalize label
        normalized_label = self._normalize_label(prediction.label)

        return SentimentResult(
            text=text,
            sentiment=normalized_label,
            confidence=prediction.score,
            raw_label=prediction.label,
            raw_score=prediction.score,
            all_scores=prediction.all_scores,
            model_name=self._model_name,
            processing_time_ms=prediction.processing_time_ms,
        )

    async def analyze_batch(
        self,
        texts: List[str],
        return_all_scores: bool = True,
        batch_size: Optional[int] = None,
    ) -> BatchSentimentResult:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of input texts
            return_all_scores: Include scores for all labels
            batch_size: Override default batch size

        Returns:
            BatchSentimentResult with aggregate statistics

        Raises:
            ValidationError: If input is invalid
            ModelInferenceError: If analysis fails

        Example:
            >>> result = await analyzer.analyze_batch([
            ...     "Great product!",
            ...     "Terrible service.",
            ...     "It's okay."
            ... ])
            >>> print(result.positive_count, result.negative_count)
            1 1
        """
        self._ensure_initialized()

        if not texts:
            return BatchSentimentResult(results=[])

        if not isinstance(texts, (list, tuple)):
            raise ValidationError("texts must be a list or tuple")

        # Validate and filter texts
        valid_texts = []
        text_indices = []
        for i, text in enumerate(texts):
            try:
                valid_text = self._validate_text(text)
                valid_texts.append(valid_text)
                text_indices.append(i)
            except InvalidInputError:
                pass  # Skip invalid texts

        if not valid_texts:
            return BatchSentimentResult(results=[])

        # Run batch prediction
        predictions = await self._provider.predict_batch(
            valid_texts,
            return_all_scores=return_all_scores,
            batch_size=batch_size or self._config.batch_size,
        )

        # Build results
        results: List[SentimentResult] = []
        for idx, (text, prediction) in enumerate(zip(valid_texts, predictions)):
            normalized_label = self._normalize_label(prediction.label)

            results.append(
                SentimentResult(
                    text=text,
                    sentiment=normalized_label,
                    confidence=prediction.score,
                    raw_label=prediction.label,
                    raw_score=prediction.score,
                    all_scores=prediction.all_scores,
                    model_name=self._model_name,
                    processing_time_ms=prediction.processing_time_ms,
                )
            )

        return BatchSentimentResult(results=results)

    async def get_quick_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Quick sentiment analysis returning only label and score.

        Args:
            text: Input text

        Returns:
            Tuple of (sentiment_label, confidence_score)

        Example:
            >>> label, score = await analyzer.get_quick_sentiment("Great!")
            >>> print(label, score)
            positive 0.95
        """
        result = await self.analyze(text, return_all_scores=False)
        return result.sentiment.value, result.confidence

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._provider:
            return self._provider.get_model_info()
        return {"model_name": self._model_name, "loaded": False}

    @property
    def model_name(self) -> str:
        """Get model name."""
        return self._model_name

    @property
    def classification_mode(self) -> SentimentClass:
        """Get classification mode."""
        return self._classification_mode

    @property
    def is_initialized(self) -> bool:
        """Check if analyzer is initialized."""
        return self._initialized


# Convenience functions for quick analysis


async def analyze_sentiment(
    text: str,
    model_name: Optional[str] = None,
) -> SentimentResult:
    """
    Analyze sentiment of a single text (convenience function).

    Creates a temporary analyzer, analyzes text, and cleans up.

    Args:
        text: Input text
        model_name: Optional model name override

    Returns:
        SentimentResult

    Example:
        >>> result = await analyze_sentiment("Great product!")
        >>> print(result.sentiment)
        positive
    """
    async with SentimentAnalyzer(model_name=model_name) as analyzer:
        return await analyzer.analyze(text)


async def analyze_sentiment_batch(
    texts: List[str],
    model_name: Optional[str] = None,
) -> BatchSentimentResult:
    """
    Analyze sentiment of multiple texts (convenience function).

    Creates a temporary analyzer, analyzes texts, and cleans up.

    Args:
        texts: List of input texts
        model_name: Optional model name override

    Returns:
        BatchSentimentResult

    Example:
        >>> result = await analyze_sentiment_batch(["Great!", "Bad!"])
        >>> print(result.positive_count)
        1
    """
    async with SentimentAnalyzer(model_name=model_name) as analyzer:
        return await analyzer.analyze_batch(texts)


def analyze_sentiment_sync(
    text: str,
    model_name: Optional[str] = None,
) -> SentimentResult:
    """
    Synchronous wrapper for sentiment analysis.

    Args:
        text: Input text
        model_name: Optional model name override

    Returns:
        SentimentResult

    Example:
        >>> result = analyze_sentiment_sync("Great product!")
        >>> print(result.sentiment)
        positive
    """
    return asyncio.get_event_loop().run_until_complete(
        analyze_sentiment(text, model_name)
    )


def analyze_sentiment_batch_sync(
    texts: List[str],
    model_name: Optional[str] = None,
) -> BatchSentimentResult:
    """
    Synchronous wrapper for batch sentiment analysis.

    Args:
        texts: List of input texts
        model_name: Optional model name override

    Returns:
        BatchSentimentResult
    """
    return asyncio.get_event_loop().run_until_complete(
        analyze_sentiment_batch(texts, model_name)
    )
