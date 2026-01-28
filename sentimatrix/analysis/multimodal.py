"""
Sentimatrix Multi-Modal Analysis

Provides sentiment analysis for non-text inputs:
- Audio: Transcribe and analyze
- Image: Caption and analyze
- Video: Process frames/audio and analyze

Example:
    >>> from sentimatrix.analysis.multimodal import MultiModalAnalyzer
    >>>
    >>> async with MultiModalAnalyzer() as analyzer:
    ...     # Analyze audio
    ...     result = await analyzer.analyze_audio("speech.mp3")
    ...     print(result.sentiment)
    ...
    ...     # Analyze image
    ...     result = await analyzer.analyze_image("product.jpg")
    ...     print(result.caption, result.sentiment)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from sentimatrix.core.logger import get_logger
from sentimatrix.input.handlers import (
    AudioHandler,
    AudioEngine,
    ImageHandler,
    ImageModel,
    VideoHandler,
    TranscriptionResult,
    CaptionResult,
    VideoFrameResult,
)
from sentimatrix.analysis.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
)
from sentimatrix.analysis.emotion import (
    EmotionDetector,
    EmotionResult,
)

logger = get_logger(__name__)


# ============================================================================
# Data Classes
# ============================================================================


class FusionStrategy(str, Enum):
    """Strategies for combining multi-modal results."""
    LATE = "late"  # Analyze separately, merge results
    WEIGHTED = "weighted"  # Weight by confidence
    DOMINANT = "dominant"  # Use highest confidence


@dataclass
class AudioAnalysisResult:
    """Result of audio sentiment analysis."""

    transcription: TranscriptionResult
    sentiment: Optional[SentimentResult] = None
    emotions: Optional[EmotionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transcription": self.transcription.to_dict(),
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "emotions": self.emotions.to_dict() if self.emotions else None,
            "metadata": self.metadata,
        }


@dataclass
class ImageAnalysisResult:
    """Result of image sentiment analysis."""

    caption: CaptionResult
    sentiment: Optional[SentimentResult] = None
    emotions: Optional[EmotionResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "caption": self.caption.to_dict(),
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "emotions": self.emotions.to_dict() if self.emotions else None,
            "metadata": self.metadata,
        }


@dataclass
class VideoAnalysisResult:
    """Result of video sentiment analysis."""

    frames_analyzed: int
    duration_seconds: float
    audio_result: Optional[AudioAnalysisResult] = None
    frame_sentiments: List[SentimentResult] = field(default_factory=list)
    combined_sentiment: Optional[SentimentResult] = None
    sentiment_timeline: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frames_analyzed": self.frames_analyzed,
            "duration_seconds": self.duration_seconds,
            "audio_result": self.audio_result.to_dict() if self.audio_result else None,
            "frame_sentiments": [s.to_dict() for s in self.frame_sentiments],
            "combined_sentiment": self.combined_sentiment.to_dict() if self.combined_sentiment else None,
            "sentiment_timeline": self.sentiment_timeline,
            "metadata": self.metadata,
        }


@dataclass
class MultiModalResult:
    """Combined result from multiple modalities."""

    input_type: str
    modalities_analyzed: List[str]
    text_result: Optional[SentimentResult] = None
    audio_result: Optional[AudioAnalysisResult] = None
    image_result: Optional[ImageAnalysisResult] = None
    video_result: Optional[VideoAnalysisResult] = None
    combined_sentiment: Optional[SentimentResult] = None
    combined_emotions: Optional[EmotionResult] = None
    fusion_method: str = "late"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input_type": self.input_type,
            "modalities_analyzed": self.modalities_analyzed,
            "text_result": self.text_result.to_dict() if self.text_result else None,
            "audio_result": self.audio_result.to_dict() if self.audio_result else None,
            "image_result": self.image_result.to_dict() if self.image_result else None,
            "video_result": self.video_result.to_dict() if self.video_result else None,
            "combined_sentiment": self.combined_sentiment.to_dict() if self.combined_sentiment else None,
            "combined_emotions": self.combined_emotions.to_dict() if self.combined_emotions else None,
            "fusion_method": self.fusion_method,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# ============================================================================
# Multi-Modal Analyzer
# ============================================================================


class MultiModalAnalyzer:
    """
    Multi-modal sentiment analyzer.

    Combines audio, image, and video analysis with text-based
    sentiment analysis for comprehensive multi-modal understanding.

    Example:
        >>> analyzer = MultiModalAnalyzer(
        ...     audio_engine="groq_whisper",
        ...     image_model="gpt4v",
        ... )
        >>> await analyzer.initialize()
        >>>
        >>> # Analyze audio
        >>> result = await analyzer.analyze_audio("interview.mp3")
        >>> print(f"Sentiment: {result.sentiment.sentiment}")
        >>>
        >>> # Analyze image
        >>> result = await analyzer.analyze_image("product.jpg")
        >>> print(f"Caption: {result.caption.caption}")
    """

    def __init__(
        self,
        audio_engine: Union[str, AudioEngine] = AudioEngine.WHISPER_LOCAL,
        audio_model: str = "base",
        image_model: Union[str, ImageModel] = ImageModel.GPT4V,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        fusion_strategy: FusionStrategy = FusionStrategy.WEIGHTED,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize multi-modal analyzer.

        Args:
            audio_engine: Audio transcription engine
            audio_model: Audio model size (for local Whisper)
            image_model: Image captioning model
            api_key: Default API key for all providers
            openai_api_key: OpenAI API key
            groq_api_key: Groq API key
            anthropic_api_key: Anthropic API key
            google_api_key: Google API key
            fusion_strategy: How to combine multi-modal results
            weights: Modality weights for fusion (text, audio, image)
        """
        self._audio_engine = audio_engine
        self._audio_model = audio_model
        self._image_model = image_model

        # API keys
        self._api_key = api_key
        self._openai_api_key = openai_api_key or api_key
        self._groq_api_key = groq_api_key or api_key
        self._anthropic_api_key = anthropic_api_key or api_key
        self._google_api_key = google_api_key or api_key

        self._fusion_strategy = fusion_strategy
        self._weights = weights or {"text": 0.5, "audio": 0.3, "image": 0.2}

        # Handlers (lazy initialized)
        self._audio_handler: Optional[AudioHandler] = None
        self._image_handler: Optional[ImageHandler] = None
        self._video_handler: Optional[VideoHandler] = None

        # Analyzers
        self._sentiment_analyzer: Optional[SentimentAnalyzer] = None
        self._emotion_detector: Optional[EmotionDetector] = None

        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all handlers and analyzers."""
        if self._initialized:
            return

        # Initialize sentiment analyzer
        self._sentiment_analyzer = SentimentAnalyzer()
        await self._sentiment_analyzer.initialize()

        # Initialize emotion detector
        self._emotion_detector = EmotionDetector()
        await self._emotion_detector.initialize()

        self._initialized = True
        logger.info("MultiModalAnalyzer initialized")

    async def _ensure_audio_handler(self) -> AudioHandler:
        """Ensure audio handler is initialized."""
        if self._audio_handler is None:
            # Determine API key based on engine
            api_key = None
            if self._audio_engine in (AudioEngine.WHISPER_API, AudioEngine.OPENAI_WHISPER):
                api_key = self._openai_api_key
            elif self._audio_engine == AudioEngine.GROQ_WHISPER:
                api_key = self._groq_api_key

            self._audio_handler = AudioHandler(
                engine=self._audio_engine,
                model=self._audio_model,
                api_key=api_key,
            )
            await self._audio_handler.initialize()

        return self._audio_handler

    async def _ensure_image_handler(self) -> ImageHandler:
        """Ensure image handler is initialized."""
        if self._image_handler is None:
            # Determine API key based on model
            api_key = None
            if self._image_model == ImageModel.GPT4V:
                api_key = self._openai_api_key
            elif self._image_model == ImageModel.CLAUDE_VISION:
                api_key = self._anthropic_api_key
            elif self._image_model == ImageModel.GEMINI_VISION:
                api_key = self._google_api_key

            self._image_handler = ImageHandler(
                model=self._image_model,
                api_key=api_key,
                include_ocr=True,
            )
            await self._image_handler.initialize()

        return self._image_handler

    async def _ensure_video_handler(self) -> VideoHandler:
        """Ensure video handler is initialized."""
        if self._video_handler is None:
            self._video_handler = VideoHandler()
            await self._video_handler.initialize()

        return self._video_handler

    async def close(self) -> None:
        """Close all handlers and cleanup resources."""
        if self._audio_handler:
            await self._audio_handler.close()
        if self._image_handler:
            await self._image_handler.close()
        if self._video_handler:
            await self._video_handler.close()
        if self._sentiment_analyzer:
            await self._sentiment_analyzer.close()
        if self._emotion_detector:
            await self._emotion_detector.close()

        self._audio_handler = None
        self._image_handler = None
        self._video_handler = None
        self._sentiment_analyzer = None
        self._emotion_detector = None
        self._initialized = False

    async def __aenter__(self) -> "MultiModalAnalyzer":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

    # ========================================================================
    # Audio Analysis
    # ========================================================================

    async def analyze_audio(
        self,
        audio_input: Union[str, bytes],
        language: Optional[str] = None,
        include_emotions: bool = True,
        include_timestamps: bool = False,
    ) -> AudioAnalysisResult:
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
            include_timestamps: Include segment timestamps

        Returns:
            AudioAnalysisResult with transcription and sentiment
        """
        if not self._initialized:
            await self.initialize()

        handler = await self._ensure_audio_handler()

        # Step 1: Transcribe
        logger.debug(f"Transcribing audio...")
        transcription = await handler.transcribe(
            audio_input,
            language=language,
            include_timestamps=include_timestamps,
        )

        # Step 2: Analyze sentiment
        sentiment = None
        if transcription.text:
            sentiment = await self._sentiment_analyzer.analyze(transcription.text)

        # Step 3: Detect emotions
        emotions = None
        if include_emotions and transcription.text:
            emotions = await self._emotion_detector.detect(transcription.text)

        return AudioAnalysisResult(
            transcription=transcription,
            sentiment=sentiment,
            emotions=emotions,
            metadata={
                "engine": str(self._audio_engine),
                "model": self._audio_model,
                "language": transcription.language,
            },
        )

    async def transcribe(
        self,
        audio_input: Union[str, bytes],
        language: Optional[str] = None,
        include_timestamps: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio without sentiment analysis.

        Args:
            audio_input: Audio file path or bytes
            language: Language code
            include_timestamps: Include segment timestamps

        Returns:
            TranscriptionResult
        """
        handler = await self._ensure_audio_handler()
        return await handler.transcribe(
            audio_input,
            language=language,
            include_timestamps=include_timestamps,
        )

    # ========================================================================
    # Image Analysis
    # ========================================================================

    async def analyze_image(
        self,
        image_input: Union[str, bytes],
        prompt: Optional[str] = None,
        detailed: bool = False,
        include_emotions: bool = True,
    ) -> ImageAnalysisResult:
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

        Returns:
            ImageAnalysisResult with caption and sentiment
        """
        if not self._initialized:
            await self.initialize()

        handler = await self._ensure_image_handler()

        # Step 1: Generate caption
        logger.debug(f"Generating image caption...")

        # Use sentiment-focused prompt if none provided
        if prompt is None:
            prompt = (
                "Describe this image focusing on the emotional content, "
                "mood, and any text visible. What sentiment does it convey?"
            )

        caption = await handler.caption(
            image_input,
            prompt=prompt,
            detailed=detailed,
        )

        # Step 2: Analyze sentiment of caption
        sentiment = None
        if caption.caption:
            sentiment = await self._sentiment_analyzer.analyze(caption.caption)

        # Step 3: Detect emotions
        emotions = None
        if include_emotions and caption.caption:
            emotions = await self._emotion_detector.detect(caption.caption)

        return ImageAnalysisResult(
            caption=caption,
            sentiment=sentiment,
            emotions=emotions,
            metadata={
                "model": str(self._image_model),
                "detailed": detailed,
                "text_detected": caption.text_detected,
            },
        )

    async def caption_image(
        self,
        image_input: Union[str, bytes],
        prompt: Optional[str] = None,
        detailed: bool = False,
    ) -> CaptionResult:
        """
        Generate image caption without sentiment analysis.

        Args:
            image_input: Image file path or bytes
            prompt: Custom prompt
            detailed: Generate detailed caption

        Returns:
            CaptionResult
        """
        handler = await self._ensure_image_handler()
        return await handler.caption(
            image_input,
            prompt=prompt,
            detailed=detailed,
        )

    # ========================================================================
    # Video Analysis
    # ========================================================================

    async def analyze_video(
        self,
        video_path: str,
        analyze_audio: bool = True,
        analyze_frames: bool = True,
        max_frames: int = 10,
        sample_rate: float = 0.5,
    ) -> VideoAnalysisResult:
        """
        Analyze sentiment from video input.

        Pipeline:
        1. Extract key frames
        2. Optionally extract and analyze audio
        3. Analyze frame captions for sentiment
        4. Combine results using fusion strategy

        Args:
            video_path: Path to video file
            analyze_audio: Also analyze audio track
            analyze_frames: Analyze extracted frames
            max_frames: Maximum frames to analyze
            sample_rate: Frames per second to extract

        Returns:
            VideoAnalysisResult with combined analysis
        """
        if not self._initialized:
            await self.initialize()

        video_handler = await self._ensure_video_handler()

        # Extract frames
        logger.debug(f"Extracting frames from video...")
        frame_result = await video_handler.extract_frames(
            video_path,
            max_frames=max_frames,
            extract_audio=analyze_audio,
        )

        # Analyze audio if available
        audio_result = None
        if analyze_audio and frame_result.audio_path:
            try:
                audio_result = await self.analyze_audio(
                    frame_result.audio_path,
                    include_emotions=True,
                )
            except Exception as e:
                logger.warning(f"Audio analysis failed: {e}")

        # Analyze frames
        frame_sentiments: List[SentimentResult] = []
        sentiment_timeline: List[Dict[str, Any]] = []

        if analyze_frames and frame_result.frames:
            image_handler = await self._ensure_image_handler()

            for i, frame_bytes in enumerate(frame_result.frames[:max_frames]):
                try:
                    # Generate brief caption
                    caption = await image_handler.caption(
                        frame_bytes,
                        prompt="Describe this video frame briefly. What's happening? What's the mood?",
                        detailed=False,
                    )

                    if caption.caption:
                        sentiment = await self._sentiment_analyzer.analyze(caption.caption)
                        frame_sentiments.append(sentiment)

                        # Calculate approximate timestamp
                        timestamp = (i / len(frame_result.frames)) * frame_result.duration_seconds

                        sentiment_timeline.append({
                            "frame_index": i,
                            "timestamp": timestamp,
                            "sentiment": sentiment.sentiment,
                            "confidence": sentiment.confidence,
                            "caption": caption.caption[:100],
                        })
                except Exception as e:
                    logger.warning(f"Frame {i} analysis failed: {e}")

        # Combine results
        combined_sentiment = self._fuse_sentiments(
            frame_sentiments,
            audio_result.sentiment if audio_result else None,
        )

        return VideoAnalysisResult(
            frames_analyzed=len(frame_sentiments),
            duration_seconds=frame_result.duration_seconds,
            audio_result=audio_result,
            frame_sentiments=frame_sentiments,
            combined_sentiment=combined_sentiment,
            sentiment_timeline=sentiment_timeline,
            metadata={
                "fps": frame_result.fps,
                "width": frame_result.width,
                "height": frame_result.height,
                "total_frames_extracted": frame_result.frame_count,
            },
        )

    # ========================================================================
    # Fusion Methods
    # ========================================================================

    def _fuse_sentiments(
        self,
        frame_sentiments: List[SentimentResult],
        audio_sentiment: Optional[SentimentResult],
    ) -> Optional[SentimentResult]:
        """Fuse multiple sentiment results into one."""
        all_sentiments: List[SentimentResult] = list(frame_sentiments)
        if audio_sentiment:
            all_sentiments.append(audio_sentiment)

        if not all_sentiments:
            return None

        if self._fusion_strategy == FusionStrategy.DOMINANT:
            # Return highest confidence
            return max(all_sentiments, key=lambda s: s.confidence)

        elif self._fusion_strategy == FusionStrategy.WEIGHTED:
            # Weighted average
            sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
            total_weight = 0.0

            for sentiment in all_sentiments:
                weight = sentiment.confidence
                # Get string sentiment value
                sent_str = str(sentiment.sentiment.value) if hasattr(sentiment.sentiment, 'value') else str(sentiment.sentiment)
                if sent_str in sentiment_scores:
                    sentiment_scores[sent_str] += weight
                total_weight += weight

            if total_weight > 0:
                for key in sentiment_scores:
                    sentiment_scores[key] /= total_weight

            # Find dominant sentiment
            dominant = max(sentiment_scores.items(), key=lambda x: x[1])

            # Calculate average confidence
            avg_confidence = sum(s.confidence for s in all_sentiments) / len(all_sentiments)

            return SentimentResult(
                text="[combined]",
                sentiment=dominant[0],
                confidence=avg_confidence,
                all_scores=sentiment_scores,
            )

        else:  # LATE fusion - average
            # Count sentiments
            sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
            total_confidence = 0.0

            for sentiment in all_sentiments:
                # Get string sentiment value
                sent_str = str(sentiment.sentiment.value) if hasattr(sentiment.sentiment, 'value') else str(sentiment.sentiment)
                if sent_str in sentiment_counts:
                    sentiment_counts[sent_str] += 1
                total_confidence += sentiment.confidence

            # Majority vote
            dominant = max(sentiment_counts.items(), key=lambda x: x[1])
            avg_confidence = total_confidence / len(all_sentiments)

            return SentimentResult(
                text="[combined]",
                sentiment=dominant[0],
                confidence=avg_confidence,
                all_scores={k: v / len(all_sentiments) for k, v in sentiment_counts.items()},
            )

    async def analyze_multimodal(
        self,
        text: Optional[str] = None,
        audio: Optional[Union[str, bytes]] = None,
        image: Optional[Union[str, bytes]] = None,
    ) -> MultiModalResult:
        """
        Analyze multiple modalities and combine results.

        Args:
            text: Text input
            audio: Audio file path or bytes
            image: Image file path or bytes

        Returns:
            MultiModalResult with combined analysis
        """
        if not self._initialized:
            await self.initialize()

        modalities = []
        text_result = None
        audio_result = None
        image_result = None

        # Analyze text
        if text:
            modalities.append("text")
            text_result = await self._sentiment_analyzer.analyze(text)

        # Analyze audio
        if audio:
            modalities.append("audio")
            audio_result = await self.analyze_audio(audio)

        # Analyze image
        if image:
            modalities.append("image")
            image_result = await self.analyze_image(image)

        # Combine sentiments
        all_sentiments = []
        weights = []

        if text_result:
            all_sentiments.append(text_result)
            weights.append(self._weights.get("text", 0.5))

        if audio_result and audio_result.sentiment:
            all_sentiments.append(audio_result.sentiment)
            weights.append(self._weights.get("audio", 0.3))

        if image_result and image_result.sentiment:
            all_sentiments.append(image_result.sentiment)
            weights.append(self._weights.get("image", 0.2))

        combined_sentiment = self._weighted_fusion(all_sentiments, weights)

        # Calculate overall confidence
        confidence = 0.0
        if all_sentiments:
            confidence = sum(s.confidence * w for s, w in zip(all_sentiments, weights))
            confidence /= sum(weights)

        return MultiModalResult(
            input_type="multimodal" if len(modalities) > 1 else modalities[0] if modalities else "none",
            modalities_analyzed=modalities,
            text_result=text_result,
            audio_result=audio_result,
            image_result=image_result,
            combined_sentiment=combined_sentiment,
            fusion_method=str(self._fusion_strategy.value),
            confidence=confidence,
        )

    def _weighted_fusion(
        self,
        sentiments: List[SentimentResult],
        weights: List[float],
    ) -> Optional[SentimentResult]:
        """Fuse sentiments with explicit weights."""
        if not sentiments:
            return None

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        sentiment_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

        for sentiment, weight in zip(sentiments, normalized_weights):
            if sentiment.all_scores:
                for label, score in sentiment.all_scores.items():
                    if label in sentiment_scores:
                        sentiment_scores[label] += score * weight
            else:
                # Get string sentiment value
                sent_str = str(sentiment.sentiment.value) if hasattr(sentiment.sentiment, 'value') else str(sentiment.sentiment)
                if sent_str in sentiment_scores:
                    sentiment_scores[sent_str] += weight

        dominant = max(sentiment_scores.items(), key=lambda x: x[1])
        avg_confidence = sum(s.confidence * w for s, w in zip(sentiments, normalized_weights))

        return SentimentResult(
            text="[combined]",
            sentiment=dominant[0],
            confidence=avg_confidence,
            all_scores=sentiment_scores,
        )


# ============================================================================
# Convenience Functions
# ============================================================================


async def analyze_audio_sentiment(
    audio_input: Union[str, bytes],
    engine: str = "whisper",
    include_emotions: bool = True,
) -> AudioAnalysisResult:
    """
    Quick audio sentiment analysis.

    Args:
        audio_input: Audio file path or bytes
        engine: Transcription engine
        include_emotions: Include emotion detection

    Returns:
        AudioAnalysisResult
    """
    async with MultiModalAnalyzer(audio_engine=engine) as analyzer:
        return await analyzer.analyze_audio(audio_input, include_emotions=include_emotions)


async def analyze_image_sentiment(
    image_input: Union[str, bytes],
    model: str = "gpt4v",
    include_emotions: bool = True,
) -> ImageAnalysisResult:
    """
    Quick image sentiment analysis.

    Args:
        image_input: Image file path or bytes
        model: Captioning model
        include_emotions: Include emotion detection

    Returns:
        ImageAnalysisResult
    """
    async with MultiModalAnalyzer(image_model=model) as analyzer:
        return await analyzer.analyze_image(image_input, include_emotions=include_emotions)


async def analyze_video_sentiment(
    video_path: str,
    analyze_audio: bool = True,
    max_frames: int = 10,
) -> VideoAnalysisResult:
    """
    Quick video sentiment analysis.

    Args:
        video_path: Path to video file
        analyze_audio: Also analyze audio track
        max_frames: Maximum frames to analyze

    Returns:
        VideoAnalysisResult
    """
    async with MultiModalAnalyzer() as analyzer:
        return await analyzer.analyze_video(
            video_path,
            analyze_audio=analyze_audio,
            max_frames=max_frames,
        )
