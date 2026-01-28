"""
Sentimatrix Analysis Module

Contains analysis implementations:
- Sentiment analysis
- Emotion detection
- Multi-modal analysis (audio, image, video)
- Aspect-based analysis
"""

from sentimatrix.analysis.sentiment import (
    SentimentAnalyzer,
    SentimentResult,
    BatchSentimentResult,
    SentimentLabel,
    SentimentClass,
    analyze_sentiment,
    analyze_sentiment_batch,
    analyze_sentiment_sync,
    analyze_sentiment_batch_sync,
)
from sentimatrix.analysis.emotion import (
    EmotionDetector,
    EmotionResult,
    BatchEmotionResult,
    EmotionScore,
    EmotionCategory,
    EkmanEmotion,
    PlutchikEmotion,
    EmotionMode,
    detect_emotions,
    detect_emotions_batch,
    detect_emotions_sync,
    detect_emotions_batch_sync,
)
from sentimatrix.analysis.multimodal import (
    MultiModalAnalyzer,
    FusionStrategy,
    AudioAnalysisResult,
    ImageAnalysisResult,
    VideoAnalysisResult,
    MultiModalResult,
    analyze_audio_sentiment,
    analyze_image_sentiment,
    analyze_video_sentiment,
)

__all__ = [
    # Sentiment
    "SentimentAnalyzer",
    "SentimentResult",
    "BatchSentimentResult",
    "SentimentLabel",
    "SentimentClass",
    "analyze_sentiment",
    "analyze_sentiment_batch",
    "analyze_sentiment_sync",
    "analyze_sentiment_batch_sync",
    # Emotion
    "EmotionDetector",
    "EmotionResult",
    "BatchEmotionResult",
    "EmotionScore",
    "EmotionCategory",
    "EkmanEmotion",
    "PlutchikEmotion",
    "EmotionMode",
    "detect_emotions",
    "detect_emotions_batch",
    "detect_emotions_sync",
    "detect_emotions_batch_sync",
    # Multi-modal
    "MultiModalAnalyzer",
    "FusionStrategy",
    "AudioAnalysisResult",
    "ImageAnalysisResult",
    "VideoAnalysisResult",
    "MultiModalResult",
    "analyze_audio_sentiment",
    "analyze_image_sentiment",
    "analyze_video_sentiment",
]
