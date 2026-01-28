"""
Sentimatrix Input Module

Contains input handlers for various data formats:
- Text input
- File input
- Audio input (transcription)
- Image input (captioning)
- Video input (frame extraction)
"""

from sentimatrix.input.handlers import (
    # Enums
    AudioEngine,
    ImageModel,
    # Constants
    AUDIO_FORMATS,
    IMAGE_FORMATS,
    VIDEO_FORMATS,
    # Data classes
    TranscriptionResult,
    CaptionResult,
    VideoFrameResult,
    # Handlers
    BaseInputHandler,
    AudioHandler,
    ImageHandler,
    VideoHandler,
    # Factory functions
    get_audio_handler,
    get_image_handler,
    get_video_handler,
)

__all__ = [
    # Enums
    "AudioEngine",
    "ImageModel",
    # Constants
    "AUDIO_FORMATS",
    "IMAGE_FORMATS",
    "VIDEO_FORMATS",
    # Data classes
    "TranscriptionResult",
    "CaptionResult",
    "VideoFrameResult",
    # Handlers
    "BaseInputHandler",
    "AudioHandler",
    "ImageHandler",
    "VideoHandler",
    # Factory functions
    "get_audio_handler",
    "get_image_handler",
    "get_video_handler",
]
