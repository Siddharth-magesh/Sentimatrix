"""
Sentimatrix Emotion Detection Module

Provides comprehensive emotion detection functionality including:
- GoEmotions (28 emotions)
- Ekman's 6 basic emotions
- Multi-label classification
- Top-k emotion selection
- Emotion intensity analysis

Example:
    >>> detector = EmotionDetector()
    >>> await detector.initialize()
    >>> result = await detector.detect("I'm so happy today!")
    >>> print(result.primary_emotion, result.emotions)
    joy [{'label': 'joy', 'score': 0.95}, ...]
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sentimatrix.core.config import ModelConfig, SentimatrixConfig
from sentimatrix.core.exceptions import (
    InvalidInputError,
    ModelInferenceError,
    SentimatrixError,
    ValidationError,
)
from sentimatrix.providers.base import PredictionResult
from sentimatrix.providers.models.huggingface import (
    EmotionModelProvider,
    ModelType,
)


class EmotionCategory(str, Enum):
    """
    GoEmotions 28 emotion categories.

    Based on the Google GoEmotions dataset.
    """

    ADMIRATION = "admiration"
    AMUSEMENT = "amusement"
    ANGER = "anger"
    ANNOYANCE = "annoyance"
    APPROVAL = "approval"
    CARING = "caring"
    CONFUSION = "confusion"
    CURIOSITY = "curiosity"
    DESIRE = "desire"
    DISAPPOINTMENT = "disappointment"
    DISAPPROVAL = "disapproval"
    DISGUST = "disgust"
    EMBARRASSMENT = "embarrassment"
    EXCITEMENT = "excitement"
    FEAR = "fear"
    GRATITUDE = "gratitude"
    GRIEF = "grief"
    JOY = "joy"
    LOVE = "love"
    NERVOUSNESS = "nervousness"
    NEUTRAL = "neutral"
    OPTIMISM = "optimism"
    PRIDE = "pride"
    REALIZATION = "realization"
    RELIEF = "relief"
    REMORSE = "remorse"
    SADNESS = "sadness"
    SURPRISE = "surprise"


class EkmanEmotion(str, Enum):
    """
    Ekman's 6 basic emotions.

    These are universal emotions recognized across cultures.
    """

    ANGER = "anger"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    SADNESS = "sadness"
    SURPRISE = "surprise"


class PlutchikEmotion(str, Enum):
    """
    Plutchik's 8 primary emotions.

    Based on Plutchik's wheel of emotions.
    """

    ANGER = "anger"
    ANTICIPATION = "anticipation"
    DISGUST = "disgust"
    FEAR = "fear"
    JOY = "joy"
    SADNESS = "sadness"
    SURPRISE = "surprise"
    TRUST = "trust"


class EmotionMode(str, Enum):
    """Emotion detection modes."""

    SINGLE_LABEL = "single_label"  # Return only top emotion
    MULTI_LABEL = "multi_label"  # Return all emotions above threshold
    TOP_K = "top_k"  # Return top-k emotions


# Mapping from GoEmotions to Ekman's basic emotions
GOEMOTIONS_TO_EKMAN: Dict[str, EkmanEmotion] = {
    "anger": EkmanEmotion.ANGER,
    "annoyance": EkmanEmotion.ANGER,
    "disapproval": EkmanEmotion.ANGER,
    "disgust": EkmanEmotion.DISGUST,
    "fear": EkmanEmotion.FEAR,
    "nervousness": EkmanEmotion.FEAR,
    "joy": EkmanEmotion.JOY,
    "amusement": EkmanEmotion.JOY,
    "excitement": EkmanEmotion.JOY,
    "gratitude": EkmanEmotion.JOY,
    "love": EkmanEmotion.JOY,
    "optimism": EkmanEmotion.JOY,
    "relief": EkmanEmotion.JOY,
    "pride": EkmanEmotion.JOY,
    "admiration": EkmanEmotion.JOY,
    "approval": EkmanEmotion.JOY,
    "caring": EkmanEmotion.JOY,
    "desire": EkmanEmotion.JOY,
    "sadness": EkmanEmotion.SADNESS,
    "disappointment": EkmanEmotion.SADNESS,
    "embarrassment": EkmanEmotion.SADNESS,
    "grief": EkmanEmotion.SADNESS,
    "remorse": EkmanEmotion.SADNESS,
    "surprise": EkmanEmotion.SURPRISE,
    "realization": EkmanEmotion.SURPRISE,
    "confusion": EkmanEmotion.SURPRISE,
    "curiosity": EkmanEmotion.SURPRISE,
}

# Emotion valence (positive/negative/neutral)
EMOTION_VALENCE: Dict[str, str] = {
    "admiration": "positive",
    "amusement": "positive",
    "approval": "positive",
    "caring": "positive",
    "desire": "positive",
    "excitement": "positive",
    "gratitude": "positive",
    "joy": "positive",
    "love": "positive",
    "optimism": "positive",
    "pride": "positive",
    "relief": "positive",
    "curiosity": "neutral",
    "realization": "neutral",
    "surprise": "neutral",
    "confusion": "neutral",
    "neutral": "neutral",
    "anger": "negative",
    "annoyance": "negative",
    "disappointment": "negative",
    "disapproval": "negative",
    "disgust": "negative",
    "embarrassment": "negative",
    "fear": "negative",
    "grief": "negative",
    "nervousness": "negative",
    "remorse": "negative",
    "sadness": "negative",
}


@dataclass
class EmotionScore:
    """
    Individual emotion with score.

    Attributes:
        label: Emotion label
        score: Confidence score (0-1)
        category: Emotion category enum value
        valence: Emotional valence (positive/negative/neutral)
        ekman_mapping: Corresponding Ekman emotion
    """

    label: str
    score: float
    category: Optional[EmotionCategory] = None
    valence: str = "neutral"
    ekman_mapping: Optional[EkmanEmotion] = None

    def __post_init__(self) -> None:
        """Initialize derived fields."""
        # Try to get category enum
        try:
            self.category = EmotionCategory(self.label.lower())
        except ValueError:
            self.category = None

        # Get valence
        self.valence = EMOTION_VALENCE.get(self.label.lower(), "neutral")

        # Get Ekman mapping
        self.ekman_mapping = GOEMOTIONS_TO_EKMAN.get(self.label.lower())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "label": self.label,
            "score": self.score,
            "category": self.category.value if self.category else None,
            "valence": self.valence,
            "ekman_mapping": self.ekman_mapping.value if self.ekman_mapping else None,
        }


@dataclass
class EmotionResult:
    """
    Result of emotion detection.

    Attributes:
        text: Original input text
        primary_emotion: Primary detected emotion
        emotions: List of detected emotions with scores
        all_scores: Raw scores for all emotions
        model_name: Model used for detection
        processing_time_ms: Processing time in milliseconds
    """

    text: str
    primary_emotion: EmotionScore
    emotions: List[EmotionScore]
    all_scores: Dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    processing_time_ms: float = 0.0

    @property
    def is_positive(self) -> bool:
        """Check if primary emotion is positive."""
        return self.primary_emotion.valence == "positive"

    @property
    def is_negative(self) -> bool:
        """Check if primary emotion is negative."""
        return self.primary_emotion.valence == "negative"

    @property
    def is_neutral(self) -> bool:
        """Check if primary emotion is neutral."""
        return self.primary_emotion.valence == "neutral"

    @property
    def ekman_emotion(self) -> Optional[EkmanEmotion]:
        """Get Ekman's basic emotion for primary emotion."""
        return self.primary_emotion.ekman_mapping

    @property
    def positive_emotions(self) -> List[EmotionScore]:
        """Get all positive emotions."""
        return [e for e in self.emotions if e.valence == "positive"]

    @property
    def negative_emotions(self) -> List[EmotionScore]:
        """Get all negative emotions."""
        return [e for e in self.emotions if e.valence == "negative"]

    @property
    def emotion_labels(self) -> List[str]:
        """Get list of emotion labels."""
        return [e.label for e in self.emotions]

    def get_ekman_distribution(self) -> Dict[str, float]:
        """
        Get emotion distribution mapped to Ekman's 6 emotions.

        Returns:
            Dictionary mapping Ekman emotions to aggregated scores
        """
        ekman_scores: Dict[str, float] = {e.value: 0.0 for e in EkmanEmotion}

        for label, score in self.all_scores.items():
            ekman = GOEMOTIONS_TO_EKMAN.get(label.lower())
            if ekman:
                ekman_scores[ekman.value] = max(ekman_scores[ekman.value], score)

        return ekman_scores

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "primary_emotion": self.primary_emotion.to_dict(),
            "emotions": [e.to_dict() for e in self.emotions],
            "all_scores": self.all_scores,
            "model_name": self.model_name,
            "processing_time_ms": self.processing_time_ms,
            "is_positive": self.is_positive,
            "is_negative": self.is_negative,
            "ekman_emotion": self.ekman_emotion.value if self.ekman_emotion else None,
            "ekman_distribution": self.get_ekman_distribution(),
        }


@dataclass
class BatchEmotionResult:
    """
    Result of batch emotion detection.

    Attributes:
        results: List of individual emotion results
        total_count: Total number of texts analyzed
        emotion_counts: Count of each primary emotion
        valence_counts: Count of positive/negative/neutral
        total_processing_time_ms: Total processing time
    """

    results: List[EmotionResult]
    total_count: int = 0
    emotion_counts: Dict[str, int] = field(default_factory=dict)
    valence_counts: Dict[str, int] = field(default_factory=dict)
    total_processing_time_ms: float = 0.0

    def __post_init__(self) -> None:
        """Calculate aggregate statistics."""
        if not self.results:
            return

        self.total_count = len(self.results)

        # Count primary emotions
        self.emotion_counts = {}
        for r in self.results:
            label = r.primary_emotion.label
            self.emotion_counts[label] = self.emotion_counts.get(label, 0) + 1

        # Count valences
        self.valence_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for r in self.results:
            valence = r.primary_emotion.valence
            self.valence_counts[valence] = self.valence_counts.get(valence, 0) + 1

        # Total processing time
        self.total_processing_time_ms = sum(r.processing_time_ms for r in self.results)

    @property
    def most_common_emotion(self) -> Tuple[str, int]:
        """Get most common primary emotion."""
        if not self.emotion_counts:
            return ("neutral", 0)
        return max(self.emotion_counts.items(), key=lambda x: x[1])

    @property
    def positive_ratio(self) -> float:
        """Get ratio of positive emotions."""
        return self.valence_counts.get("positive", 0) / self.total_count if self.total_count else 0.0

    @property
    def negative_ratio(self) -> float:
        """Get ratio of negative emotions."""
        return self.valence_counts.get("negative", 0) / self.total_count if self.total_count else 0.0

    def get_emotion_distribution(self) -> Dict[str, float]:
        """Get normalized distribution of primary emotions."""
        if not self.total_count:
            return {}
        return {
            label: count / self.total_count
            for label, count in self.emotion_counts.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_count": self.total_count,
            "emotion_counts": self.emotion_counts,
            "valence_counts": self.valence_counts,
            "most_common_emotion": self.most_common_emotion[0],
            "positive_ratio": self.positive_ratio,
            "negative_ratio": self.negative_ratio,
            "emotion_distribution": self.get_emotion_distribution(),
            "total_processing_time_ms": self.total_processing_time_ms,
            "results": [r.to_dict() for r in self.results],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get summary without individual results."""
        result = self.to_dict()
        del result["results"]
        return result


class EmotionDetector:
    """
    Emotion detection engine.

    Provides methods for detecting emotions in text using HuggingFace
    models with support for:
    - Single text analysis
    - Batch processing
    - Multiple detection modes (single-label, multi-label, top-k)
    - GoEmotions (28) and Ekman (6) emotion mappings

    Example:
        >>> detector = EmotionDetector()
        >>> await detector.initialize()
        >>> result = await detector.detect("I'm so happy!")
        >>> print(result.primary_emotion.label)
        joy
    """

    # Default models
    DEFAULT_MODELS = {
        "goemotions": "SamLowe/roberta-base-go_emotions",
        "ekman": "j-hartmann/emotion-english-distilroberta-base",
        "distilbert": "bhadresh-savani/distilbert-base-uncased-emotion",
    }

    def __init__(
        self,
        config: Optional[Union[ModelConfig, SentimatrixConfig]] = None,
        model_name: Optional[str] = None,
        mode: EmotionMode = EmotionMode.MULTI_LABEL,
        threshold: float = 0.3,
        top_k: int = 3,
    ) -> None:
        """
        Initialize emotion detector.

        Args:
            config: Model or Sentimatrix configuration
            model_name: Override model name
            mode: Detection mode (single_label, multi_label, top_k)
            threshold: Score threshold for multi_label mode
            top_k: Number of emotions for top_k mode
        """
        # Handle different config types
        if isinstance(config, SentimatrixConfig):
            self._config = config.models
        else:
            self._config = config or ModelConfig()

        self._model_name = model_name or self._config.emotion_model
        self._mode = mode
        self._threshold = threshold
        self._top_k = top_k
        self._provider: Optional[EmotionModelProvider] = None
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize the detector.

        Loads the model and prepares for detection.

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        if self._initialized:
            return

        # Create and initialize provider
        config_with_model = ModelConfig(
            emotion_model=self._model_name,
            device=self._config.device,
            batch_size=self._config.batch_size,
            max_length=self._config.max_length,
            use_quantization=self._config.use_quantization,
            cache_models=self._config.cache_models,
        )

        self._provider = EmotionModelProvider(config=config_with_model)
        await self._provider.initialize()
        self._initialized = True

    async def close(self) -> None:
        """Cleanup resources."""
        if self._provider:
            await self._provider.close()
        self._initialized = False

    async def __aenter__(self) -> "EmotionDetector":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    def _ensure_initialized(self) -> None:
        """Ensure detector is initialized."""
        if not self._initialized:
            raise SentimatrixError(
                "Detector not initialized. Call initialize() first."
            )

    def _validate_text(self, text: str) -> str:
        """Validate and preprocess input text."""
        if text is None:
            raise InvalidInputError("text", "Text cannot be None")

        if not isinstance(text, str):
            raise InvalidInputError("text", f"Text must be a string, got {type(text).__name__}")

        text = text.strip()

        if not text:
            raise InvalidInputError("text", "Text cannot be empty")

        if len(text) > 100000:
            raise InvalidInputError(
                "text",
                f"Text too long: {len(text)} characters (max 100,000)",
            )

        return text

    def _filter_emotions(
        self,
        all_scores: Dict[str, float],
    ) -> List[EmotionScore]:
        """
        Filter emotions based on mode and settings.

        Args:
            all_scores: Raw scores for all emotions

        Returns:
            Filtered list of EmotionScore objects
        """
        # Sort by score descending
        sorted_scores = sorted(
            all_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        emotions: List[EmotionScore] = []

        if self._mode == EmotionMode.SINGLE_LABEL:
            # Only top emotion
            if sorted_scores:
                label, score = sorted_scores[0]
                emotions.append(EmotionScore(label=label, score=score))

        elif self._mode == EmotionMode.TOP_K:
            # Top-k emotions
            for label, score in sorted_scores[: self._top_k]:
                emotions.append(EmotionScore(label=label, score=score))

        elif self._mode == EmotionMode.MULTI_LABEL:
            # All emotions above threshold
            for label, score in sorted_scores:
                if score >= self._threshold:
                    emotions.append(EmotionScore(label=label, score=score))

            # Ensure at least one emotion
            if not emotions and sorted_scores:
                label, score = sorted_scores[0]
                emotions.append(EmotionScore(label=label, score=score))

        return emotions

    async def detect(
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
            mode: Override detection mode
            threshold: Override threshold for multi_label mode
            top_k: Override k for top_k mode

        Returns:
            EmotionResult with detected emotions

        Raises:
            InvalidInputError: If text is invalid
            ModelInferenceError: If detection fails

        Example:
            >>> result = await detector.detect("I'm so happy!")
            >>> print(result.primary_emotion.label)
            joy
        """
        self._ensure_initialized()

        try:
            text = self._validate_text(text)
        except InvalidInputError:
            # Return neutral result for invalid input
            neutral_emotion = EmotionScore(label="neutral", score=0.0)
            return EmotionResult(
                text=text if isinstance(text, str) else "",
                primary_emotion=neutral_emotion,
                emotions=[neutral_emotion],
                all_scores={},
                model_name=self._model_name,
                processing_time_ms=0.0,
            )

        # Use overrides if provided
        original_mode = self._mode
        original_threshold = self._threshold
        original_top_k = self._top_k

        try:
            if mode is not None:
                self._mode = mode
            if threshold is not None:
                self._threshold = threshold
            if top_k is not None:
                self._top_k = top_k

            # Run prediction
            prediction = await self._provider.predict(
                text,
                return_all_scores=True,
            )

            # Filter emotions
            emotions = self._filter_emotions(prediction.all_scores)

            # Primary is first (highest score)
            primary = emotions[0] if emotions else EmotionScore(label="neutral", score=0.0)

            return EmotionResult(
                text=text,
                primary_emotion=primary,
                emotions=emotions,
                all_scores=prediction.all_scores,
                model_name=self._model_name,
                processing_time_ms=prediction.processing_time_ms,
            )

        finally:
            # Restore original settings
            self._mode = original_mode
            self._threshold = original_threshold
            self._top_k = original_top_k

    async def detect_batch(
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
            texts: List of input texts
            mode: Override detection mode
            threshold: Override threshold for multi_label mode
            top_k: Override k for top_k mode
            batch_size: Override default batch size

        Returns:
            BatchEmotionResult with aggregate statistics

        Raises:
            ValidationError: If input is invalid
            ModelInferenceError: If detection fails
        """
        self._ensure_initialized()

        if not texts:
            return BatchEmotionResult(results=[])

        if not isinstance(texts, (list, tuple)):
            raise ValidationError("texts must be a list or tuple")

        # Use overrides if provided
        original_mode = self._mode
        original_threshold = self._threshold
        original_top_k = self._top_k

        try:
            if mode is not None:
                self._mode = mode
            if threshold is not None:
                self._threshold = threshold
            if top_k is not None:
                self._top_k = top_k

            # Validate and filter texts
            valid_texts = []
            for text in texts:
                try:
                    valid_text = self._validate_text(text)
                    valid_texts.append(valid_text)
                except InvalidInputError:
                    pass

            if not valid_texts:
                return BatchEmotionResult(results=[])

            # Run batch prediction
            predictions = await self._provider.predict_batch(
                valid_texts,
                return_all_scores=True,
                batch_size=batch_size or self._config.batch_size,
            )

            # Build results
            results: List[EmotionResult] = []
            for text, prediction in zip(valid_texts, predictions):
                emotions = self._filter_emotions(prediction.all_scores)
                primary = emotions[0] if emotions else EmotionScore(label="neutral", score=0.0)

                results.append(
                    EmotionResult(
                        text=text,
                        primary_emotion=primary,
                        emotions=emotions,
                        all_scores=prediction.all_scores,
                        model_name=self._model_name,
                        processing_time_ms=prediction.processing_time_ms,
                    )
                )

            return BatchEmotionResult(results=results)

        finally:
            # Restore original settings
            self._mode = original_mode
            self._threshold = original_threshold
            self._top_k = original_top_k

    async def detect_top_k(
        self,
        text: str,
        k: int = 3,
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get top-k emotions for input text.

        Convenience method for top-k detection.

        Args:
            text: Input text
            k: Number of top emotions
            threshold: Minimum score threshold

        Returns:
            List of emotion dictionaries

        Example:
            >>> emotions = await detector.detect_top_k("I'm happy!", k=3)
            >>> print(emotions[0]["label"])
            joy
        """
        result = await self.detect(
            text,
            mode=EmotionMode.TOP_K,
            top_k=k,
            threshold=threshold,
        )
        return [e.to_dict() for e in result.emotions]

    async def detect_multi_label(
        self,
        text: str,
        threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Get all emotions above threshold.

        Convenience method for multi-label detection.

        Args:
            text: Input text
            threshold: Minimum score threshold

        Returns:
            List of emotion dictionaries

        Example:
            >>> emotions = await detector.detect_multi_label("I'm happy!", threshold=0.3)
        """
        result = await self.detect(
            text,
            mode=EmotionMode.MULTI_LABEL,
            threshold=threshold,
        )
        return [e.to_dict() for e in result.emotions]

    async def detect_ekman(
        self,
        text: str,
    ) -> Dict[str, float]:
        """
        Get Ekman's 6 basic emotions distribution.

        Maps detected emotions to Ekman's basic emotions.

        Args:
            text: Input text

        Returns:
            Dictionary mapping Ekman emotions to scores

        Example:
            >>> ekman = await detector.detect_ekman("I'm happy!")
            >>> print(ekman["joy"])
            0.95
        """
        result = await self.detect(text, mode=EmotionMode.SINGLE_LABEL)
        return result.get_ekman_distribution()

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
    def mode(self) -> EmotionMode:
        """Get current detection mode."""
        return self._mode

    @property
    def threshold(self) -> float:
        """Get current threshold."""
        return self._threshold

    @property
    def top_k(self) -> int:
        """Get current top-k value."""
        return self._top_k

    @property
    def is_initialized(self) -> bool:
        """Check if detector is initialized."""
        return self._initialized


# Convenience functions for quick detection


async def detect_emotions(
    text: str,
    model_name: Optional[str] = None,
    mode: EmotionMode = EmotionMode.MULTI_LABEL,
    threshold: float = 0.3,
    top_k: int = 3,
) -> EmotionResult:
    """
    Detect emotions in text (convenience function).

    Creates a temporary detector, detects emotions, and cleans up.

    Args:
        text: Input text
        model_name: Optional model name override
        mode: Detection mode
        threshold: Threshold for multi_label mode
        top_k: K for top_k mode

    Returns:
        EmotionResult

    Example:
        >>> result = await detect_emotions("I'm so happy!")
        >>> print(result.primary_emotion.label)
        joy
    """
    async with EmotionDetector(
        model_name=model_name,
        mode=mode,
        threshold=threshold,
        top_k=top_k,
    ) as detector:
        return await detector.detect(text)


async def detect_emotions_batch(
    texts: List[str],
    model_name: Optional[str] = None,
    mode: EmotionMode = EmotionMode.MULTI_LABEL,
    threshold: float = 0.3,
    top_k: int = 3,
) -> BatchEmotionResult:
    """
    Detect emotions in multiple texts (convenience function).

    Args:
        texts: List of input texts
        model_name: Optional model name override
        mode: Detection mode
        threshold: Threshold for multi_label mode
        top_k: K for top_k mode

    Returns:
        BatchEmotionResult
    """
    async with EmotionDetector(
        model_name=model_name,
        mode=mode,
        threshold=threshold,
        top_k=top_k,
    ) as detector:
        return await detector.detect_batch(texts)


def detect_emotions_sync(
    text: str,
    model_name: Optional[str] = None,
    mode: EmotionMode = EmotionMode.MULTI_LABEL,
    threshold: float = 0.3,
    top_k: int = 3,
) -> EmotionResult:
    """
    Synchronous wrapper for emotion detection.

    Args:
        text: Input text
        model_name: Optional model name override
        mode: Detection mode
        threshold: Threshold for multi_label mode
        top_k: K for top_k mode

    Returns:
        EmotionResult
    """
    return asyncio.get_event_loop().run_until_complete(
        detect_emotions(text, model_name, mode, threshold, top_k)
    )


def detect_emotions_batch_sync(
    texts: List[str],
    model_name: Optional[str] = None,
    mode: EmotionMode = EmotionMode.MULTI_LABEL,
    threshold: float = 0.3,
    top_k: int = 3,
) -> BatchEmotionResult:
    """
    Synchronous wrapper for batch emotion detection.

    Args:
        texts: List of input texts
        model_name: Optional model name override
        mode: Detection mode
        threshold: Threshold for multi_label mode
        top_k: K for top_k mode

    Returns:
        BatchEmotionResult
    """
    return asyncio.get_event_loop().run_until_complete(
        detect_emotions_batch(texts, model_name, mode, threshold, top_k)
    )
