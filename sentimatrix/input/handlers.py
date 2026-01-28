"""
Sentimatrix Input Handlers

Provides handlers for various input types:
- TextHandler: Plain text and file input
- AudioHandler: Audio file transcription (WAV, MP3, FLAC, etc.)
- ImageHandler: Image captioning and OCR
- VideoHandler: Video frame extraction and processing

Example:
    >>> from sentimatrix.input.handlers import AudioHandler, ImageHandler
    >>>
    >>> # Audio transcription
    >>> audio_handler = AudioHandler(engine="whisper")
    >>> await audio_handler.initialize()
    >>> result = await audio_handler.transcribe("audio.mp3")
    >>> print(result.text)
    >>>
    >>> # Image captioning
    >>> image_handler = ImageHandler(model="llava")
    >>> await image_handler.initialize()
    >>> result = await image_handler.caption("image.jpg")
    >>> print(result.caption)
"""

from __future__ import annotations

import asyncio
import base64
import io
import mimetypes
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, BinaryIO

from sentimatrix.core.exceptions import (
    SentimatrixError,
    ValidationError,
    ProviderInitializationError,
)
from sentimatrix.core.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class AudioEngine(str, Enum):
    """Supported audio transcription engines."""
    WHISPER_LOCAL = "whisper"
    WHISPER_API = "whisper_api"
    GROQ_WHISPER = "groq_whisper"
    OPENAI_WHISPER = "openai_whisper"


class ImageModel(str, Enum):
    """Supported image understanding models."""
    LLAVA = "llava"
    BLIP = "blip"
    GPT4V = "gpt4v"
    CLAUDE_VISION = "claude_vision"
    GEMINI_VISION = "gemini_vision"


# Supported audio formats
AUDIO_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac"}

# Supported image formats
IMAGE_FORMATS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

# Supported video formats
VIDEO_FORMATS = {".mp4", ".avi", ".mov", ".webm", ".mkv", ".wmv"}


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: Optional[str] = None
    duration_seconds: Optional[float] = None
    segments: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    word_timestamps: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "language": self.language,
            "duration_seconds": self.duration_seconds,
            "segments": self.segments,
            "confidence": self.confidence,
            "word_timestamps": self.word_timestamps,
            "metadata": self.metadata,
        }


@dataclass
class CaptionResult:
    """Result of image captioning."""

    caption: str
    confidence: Optional[float] = None
    objects: Optional[List[str]] = None
    text_detected: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "caption": self.caption,
            "confidence": self.confidence,
            "objects": self.objects,
            "text_detected": self.text_detected,
            "metadata": self.metadata,
        }


@dataclass
class VideoFrameResult:
    """Result of video frame extraction."""

    frames: List[bytes]
    frame_count: int
    duration_seconds: float
    fps: float
    width: int
    height: int
    audio_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "frame_count": self.frame_count,
            "duration_seconds": self.duration_seconds,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "audio_path": self.audio_path,
            "metadata": self.metadata,
        }


# ============================================================================
# Base Handler
# ============================================================================


class BaseInputHandler(ABC):
    """Abstract base class for input handlers."""

    def __init__(self) -> None:
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the handler."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close and cleanup resources."""
        pass

    async def __aenter__(self) -> "BaseInputHandler":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()


# ============================================================================
# Audio Handler
# ============================================================================


class AudioHandler(BaseInputHandler):
    """
    Handler for audio input processing.

    Supports transcription using various engines:
    - Local Whisper (OpenAI's Whisper model)
    - Whisper API (OpenAI)
    - Groq Whisper (fast inference)

    Example:
        >>> handler = AudioHandler(engine="whisper", model="base")
        >>> await handler.initialize()
        >>> result = await handler.transcribe("audio.mp3")
        >>> print(result.text)
    """

    def __init__(
        self,
        engine: Union[str, AudioEngine] = AudioEngine.WHISPER_LOCAL,
        model: str = "base",
        language: Optional[str] = None,
        api_key: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """
        Initialize audio handler.

        Args:
            engine: Transcription engine to use
            model: Model size (tiny, base, small, medium, large)
            language: Language code (auto-detect if None)
            api_key: API key for cloud services
            device: Device for local models (cpu, cuda, mps)
        """
        super().__init__()
        self._engine = AudioEngine(engine) if isinstance(engine, str) else engine
        self._model_name = model
        self._language = language
        self._api_key = api_key
        self._device = device
        self._model: Any = None
        self._client: Any = None

    async def initialize(self) -> None:
        """Initialize the transcription engine."""
        if self._initialized:
            return

        if self._engine == AudioEngine.WHISPER_LOCAL:
            await self._init_whisper_local()
        elif self._engine in (AudioEngine.WHISPER_API, AudioEngine.OPENAI_WHISPER):
            await self._init_whisper_api()
        elif self._engine == AudioEngine.GROQ_WHISPER:
            await self._init_groq_whisper()
        else:
            raise ValueError(f"Unknown engine: {self._engine}")

        self._initialized = True
        logger.info(f"AudioHandler initialized with engine={self._engine}")

    async def _init_whisper_local(self) -> None:
        """Initialize local Whisper model."""
        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper package is required for local Whisper. "
                "Install it with: pip install openai-whisper"
            )

        # Determine device
        device = self._device
        if device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"

        # Load model in thread pool (CPU-intensive)
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(
            None, whisper.load_model, self._model_name, device
        )
        self._device = device

    async def _init_whisper_api(self) -> None:
        """Initialize OpenAI Whisper API client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for Whisper API. "
                "Install it with: pip install openai"
            )

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderInitializationError(
                "openai",
                "API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self._client = AsyncOpenAI(api_key=api_key)

    async def _init_groq_whisper(self) -> None:
        """Initialize Groq Whisper client."""
        try:
            from groq import AsyncGroq
        except ImportError:
            raise ImportError(
                "groq package is required for Groq Whisper. "
                "Install it with: pip install groq"
            )

        api_key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ProviderInitializationError(
                "groq",
                "API key not provided. Set GROQ_API_KEY environment variable."
            )

        self._client = AsyncGroq(api_key=api_key)

    async def close(self) -> None:
        """Close and cleanup resources."""
        if self._client and hasattr(self._client, "close"):
            await self._client.close()
        self._model = None
        self._client = None
        self._initialized = False

    def _validate_audio_file(self, file_path: str) -> None:
        """Validate audio file."""
        path = Path(file_path)

        if not path.exists():
            raise ValidationError(f"Audio file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in AUDIO_FORMATS:
            raise ValidationError(
                f"Unsupported audio format: {suffix}. "
                f"Supported formats: {', '.join(AUDIO_FORMATS)}"
            )

    async def transcribe(
        self,
        audio_input: Union[str, bytes, BinaryIO],
        language: Optional[str] = None,
        include_timestamps: bool = False,
        include_word_timestamps: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_input: Audio file path, bytes, or file-like object
            language: Language code (overrides default)
            include_timestamps: Include segment timestamps
            include_word_timestamps: Include word-level timestamps

        Returns:
            TranscriptionResult with transcribed text
        """
        if not self._initialized:
            await self.initialize()

        # Handle different input types
        if isinstance(audio_input, str):
            self._validate_audio_file(audio_input)
            file_path = audio_input
        elif isinstance(audio_input, bytes):
            # Write to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_input)
                file_path = f.name
        else:
            # File-like object
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                f.write(audio_input.read())
                file_path = f.name

        try:
            if self._engine == AudioEngine.WHISPER_LOCAL:
                return await self._transcribe_whisper_local(
                    file_path, language, include_timestamps, include_word_timestamps
                )
            elif self._engine in (AudioEngine.WHISPER_API, AudioEngine.OPENAI_WHISPER):
                return await self._transcribe_whisper_api(
                    file_path, language, include_timestamps
                )
            elif self._engine == AudioEngine.GROQ_WHISPER:
                return await self._transcribe_groq(
                    file_path, language, include_timestamps
                )
            else:
                raise ValueError(f"Unknown engine: {self._engine}")
        finally:
            # Cleanup temp file if we created one
            if not isinstance(audio_input, str) and os.path.exists(file_path):
                os.unlink(file_path)

    async def _transcribe_whisper_local(
        self,
        file_path: str,
        language: Optional[str],
        include_timestamps: bool,
        include_word_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe using local Whisper model."""
        import whisper

        effective_language = language or self._language

        # Run transcription in thread pool
        loop = asyncio.get_event_loop()
        options = {
            "language": effective_language,
            "word_timestamps": include_word_timestamps,
        }

        result = await loop.run_in_executor(
            None,
            lambda: self._model.transcribe(file_path, **options)
        )

        segments = []
        if include_timestamps and "segments" in result:
            segments = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in result["segments"]
            ]

        word_timestamps = None
        if include_word_timestamps and "segments" in result:
            word_timestamps = []
            for seg in result["segments"]:
                if "words" in seg:
                    word_timestamps.extend(seg["words"])

        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language"),
            segments=segments,
            word_timestamps=word_timestamps,
            metadata={
                "engine": "whisper_local",
                "model": self._model_name,
                "device": self._device,
            },
        )

    async def _transcribe_whisper_api(
        self,
        file_path: str,
        language: Optional[str],
        include_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe using OpenAI Whisper API."""
        effective_language = language or self._language

        with open(file_path, "rb") as audio_file:
            response_format = "verbose_json" if include_timestamps else "json"

            response = await self._client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=effective_language,
                response_format=response_format,
            )

        segments = []
        if include_timestamps and hasattr(response, "segments"):
            segments = [
                {
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                }
                for seg in response.segments
            ]

        return TranscriptionResult(
            text=response.text,
            language=getattr(response, "language", effective_language),
            duration_seconds=getattr(response, "duration", None),
            segments=segments,
            metadata={
                "engine": "whisper_api",
                "model": "whisper-1",
            },
        )

    async def _transcribe_groq(
        self,
        file_path: str,
        language: Optional[str],
        include_timestamps: bool,
    ) -> TranscriptionResult:
        """Transcribe using Groq Whisper."""
        effective_language = language or self._language

        with open(file_path, "rb") as audio_file:
            response_format = "verbose_json" if include_timestamps else "json"

            response = await self._client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language=effective_language,
                response_format=response_format,
            )

        segments = []
        if include_timestamps and hasattr(response, "segments"):
            segments = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                }
                for seg in response.segments
            ]

        return TranscriptionResult(
            text=response.text,
            language=getattr(response, "language", effective_language),
            duration_seconds=getattr(response, "duration", None),
            segments=segments,
            metadata={
                "engine": "groq_whisper",
                "model": "whisper-large-v3",
            },
        )

    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Get audio file duration in seconds."""
        try:
            import wave
            with wave.open(file_path, "rb") as audio:
                frames = audio.getnframes()
                rate = audio.getframerate()
                return frames / float(rate)
        except Exception:
            # Fallback: try pydub
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_file(file_path)
                return len(audio) / 1000.0
            except ImportError:
                return 0.0


# ============================================================================
# Image Handler
# ============================================================================


class ImageHandler(BaseInputHandler):
    """
    Handler for image input processing.

    Supports captioning using various models:
    - LLaVA (local)
    - BLIP (local)
    - GPT-4 Vision (API)
    - Claude Vision (API)
    - Gemini Vision (API)

    Example:
        >>> handler = ImageHandler(model="gpt4v", api_key="...")
        >>> await handler.initialize()
        >>> result = await handler.caption("image.jpg")
        >>> print(result.caption)
    """

    def __init__(
        self,
        model: Union[str, ImageModel] = ImageModel.GPT4V,
        api_key: Optional[str] = None,
        include_ocr: bool = False,
        max_resolution: int = 1024,
    ) -> None:
        """
        Initialize image handler.

        Args:
            model: Captioning model to use
            api_key: API key for cloud services
            include_ocr: Include text extraction (OCR)
            max_resolution: Maximum image resolution (resizes if larger)
        """
        super().__init__()
        self._model_type = ImageModel(model) if isinstance(model, str) else model
        self._api_key = api_key
        self._include_ocr = include_ocr
        self._max_resolution = max_resolution
        self._client: Any = None
        self._model: Any = None
        self._processor: Any = None

    async def initialize(self) -> None:
        """Initialize the captioning model."""
        if self._initialized:
            return

        if self._model_type == ImageModel.GPT4V:
            await self._init_gpt4v()
        elif self._model_type == ImageModel.CLAUDE_VISION:
            await self._init_claude_vision()
        elif self._model_type == ImageModel.GEMINI_VISION:
            await self._init_gemini_vision()
        elif self._model_type == ImageModel.LLAVA:
            await self._init_llava()
        elif self._model_type == ImageModel.BLIP:
            await self._init_blip()
        else:
            raise ValueError(f"Unknown model: {self._model_type}")

        self._initialized = True
        logger.info(f"ImageHandler initialized with model={self._model_type}")

    async def _init_gpt4v(self) -> None:
        """Initialize GPT-4 Vision client."""
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for GPT-4 Vision. "
                "Install it with: pip install openai"
            )

        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ProviderInitializationError(
                "openai",
                "API key not provided. Set OPENAI_API_KEY environment variable."
            )

        self._client = AsyncOpenAI(api_key=api_key)

    async def _init_claude_vision(self) -> None:
        """Initialize Claude Vision client."""
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required for Claude Vision. "
                "Install it with: pip install anthropic"
            )

        api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ProviderInitializationError(
                "anthropic",
                "API key not provided. Set ANTHROPIC_API_KEY environment variable."
            )

        self._client = AsyncAnthropic(api_key=api_key)

    async def _init_gemini_vision(self) -> None:
        """Initialize Gemini Vision client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required for Gemini Vision. "
                "Install it with: pip install google-generativeai"
            )

        api_key = self._api_key or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ProviderInitializationError(
                "gemini",
                "API key not provided. Set GOOGLE_API_KEY environment variable."
            )

        genai.configure(api_key=api_key)
        self._client = genai.GenerativeModel("gemini-1.5-pro")

    async def _init_llava(self) -> None:
        """Initialize LLaVA model (local)."""
        raise NotImplementedError(
            "Local LLaVA support requires significant setup. "
            "Consider using GPT-4 Vision or Claude Vision instead."
        )

    async def _init_blip(self) -> None:
        """Initialize BLIP model (local)."""
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required for BLIP. "
                "Install with: pip install transformers torch"
            )

        loop = asyncio.get_event_loop()

        # Load model and processor
        self._processor = await loop.run_in_executor(
            None,
            lambda: BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        )
        self._model = await loop.run_in_executor(
            None,
            lambda: BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base"
            )
        )

        # Move to appropriate device
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._model = self._model.to("mps")

    async def close(self) -> None:
        """Close and cleanup resources."""
        if self._client and hasattr(self._client, "close"):
            await self._client.close()
        self._client = None
        self._model = None
        self._processor = None
        self._initialized = False

    def _validate_image_file(self, file_path: str) -> None:
        """Validate image file."""
        path = Path(file_path)

        if not path.exists():
            raise ValidationError(f"Image file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in IMAGE_FORMATS:
            raise ValidationError(
                f"Unsupported image format: {suffix}. "
                f"Supported formats: {', '.join(IMAGE_FORMATS)}"
            )

    def _load_image(self, image_input: Union[str, bytes]) -> bytes:
        """Load image and resize if necessary."""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required: pip install Pillow")

        # Load image
        if isinstance(image_input, str):
            self._validate_image_file(image_input)
            img = Image.open(image_input)
        else:
            img = Image.open(io.BytesIO(image_input))

        # Resize if too large
        if max(img.size) > self._max_resolution:
            ratio = self._max_resolution / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Convert to RGB if necessary
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    def _encode_image_base64(self, image_bytes: bytes) -> str:
        """Encode image to base64."""
        return base64.b64encode(image_bytes).decode("utf-8")

    async def caption(
        self,
        image_input: Union[str, bytes],
        prompt: Optional[str] = None,
        detailed: bool = False,
    ) -> CaptionResult:
        """
        Generate caption for an image.

        Args:
            image_input: Image file path or bytes
            prompt: Custom prompt for captioning
            detailed: Generate detailed description

        Returns:
            CaptionResult with generated caption
        """
        if not self._initialized:
            await self.initialize()

        image_bytes = self._load_image(image_input)

        if self._model_type == ImageModel.GPT4V:
            return await self._caption_gpt4v(image_bytes, prompt, detailed)
        elif self._model_type == ImageModel.CLAUDE_VISION:
            return await self._caption_claude(image_bytes, prompt, detailed)
        elif self._model_type == ImageModel.GEMINI_VISION:
            return await self._caption_gemini(image_bytes, prompt, detailed)
        elif self._model_type == ImageModel.BLIP:
            return await self._caption_blip(image_bytes, prompt, detailed)
        else:
            raise ValueError(f"Unknown model: {self._model_type}")

    async def _caption_gpt4v(
        self,
        image_bytes: bytes,
        prompt: Optional[str],
        detailed: bool,
    ) -> CaptionResult:
        """Generate caption using GPT-4 Vision."""
        base64_image = self._encode_image_base64(image_bytes)

        if prompt:
            caption_prompt = prompt
        elif detailed:
            caption_prompt = (
                "Describe this image in detail. Include objects, colors, "
                "actions, mood, and any text visible in the image."
            )
        else:
            caption_prompt = "Describe this image briefly in one or two sentences."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                            "detail": "high" if detailed else "low",
                        },
                    },
                ],
            }
        ]

        response = await self._client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
        )

        caption = response.choices[0].message.content.strip()

        # Optionally extract text (OCR)
        text_detected = None
        if self._include_ocr:
            text_detected = await self._extract_text_gpt4v(image_bytes)

        return CaptionResult(
            caption=caption,
            text_detected=text_detected,
            metadata={
                "model": "gpt-4o",
                "detailed": detailed,
            },
        )

    async def _extract_text_gpt4v(self, image_bytes: bytes) -> Optional[str]:
        """Extract text from image using GPT-4 Vision."""
        base64_image = self._encode_image_base64(image_bytes)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all visible text from this image. Return only the text, nothing else. If no text is visible, respond with 'NO_TEXT'.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ]

        response = await self._client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
        )

        text = response.choices[0].message.content.strip()
        return None if text == "NO_TEXT" else text

    async def _caption_claude(
        self,
        image_bytes: bytes,
        prompt: Optional[str],
        detailed: bool,
    ) -> CaptionResult:
        """Generate caption using Claude Vision."""
        base64_image = self._encode_image_base64(image_bytes)

        if prompt:
            caption_prompt = prompt
        elif detailed:
            caption_prompt = (
                "Describe this image in detail. Include objects, colors, "
                "actions, mood, and any text visible in the image."
            )
        else:
            caption_prompt = "Describe this image briefly in one or two sentences."

        response = await self._client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": caption_prompt},
                    ],
                }
            ],
        )

        caption = response.content[0].text.strip()

        return CaptionResult(
            caption=caption,
            metadata={
                "model": "claude-3-5-sonnet",
                "detailed": detailed,
            },
        )

    async def _caption_gemini(
        self,
        image_bytes: bytes,
        prompt: Optional[str],
        detailed: bool,
    ) -> CaptionResult:
        """Generate caption using Gemini Vision."""
        from PIL import Image

        # Load image for Gemini
        img = Image.open(io.BytesIO(image_bytes))

        if prompt:
            caption_prompt = prompt
        elif detailed:
            caption_prompt = (
                "Describe this image in detail. Include objects, colors, "
                "actions, mood, and any text visible in the image."
            )
        else:
            caption_prompt = "Describe this image briefly in one or two sentences."

        # Run in executor (Gemini SDK is sync)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._client.generate_content([caption_prompt, img])
        )

        caption = response.text.strip()

        return CaptionResult(
            caption=caption,
            metadata={
                "model": "gemini-1.5-pro",
                "detailed": detailed,
            },
        )

    async def _caption_blip(
        self,
        image_bytes: bytes,
        prompt: Optional[str],
        detailed: bool,
    ) -> CaptionResult:
        """Generate caption using local BLIP model."""
        from PIL import Image
        import torch

        img = Image.open(io.BytesIO(image_bytes))

        loop = asyncio.get_event_loop()

        def generate():
            inputs = self._processor(img, return_tensors="pt")
            if next(self._model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model.generate(**inputs, max_new_tokens=100)

            return self._processor.decode(output[0], skip_special_tokens=True)

        caption = await loop.run_in_executor(None, generate)

        return CaptionResult(
            caption=caption,
            metadata={
                "model": "blip-base",
                "detailed": detailed,
            },
        )

    async def analyze_sentiment_context(
        self,
        image_input: Union[str, bytes],
    ) -> Dict[str, Any]:
        """
        Analyze image for sentiment-relevant context.

        Returns description focused on emotional content.
        """
        if not self._initialized:
            await self.initialize()

        prompt = (
            "Analyze this image for sentiment analysis context. Describe:\n"
            "1. Overall mood/atmosphere\n"
            "2. Emotional expressions (if people present)\n"
            "3. Colors and their emotional connotations\n"
            "4. Any text or symbols visible\n"
            "5. Overall sentiment (positive, negative, neutral)\n"
            "Be concise but thorough."
        )

        result = await self.caption(image_input, prompt=prompt, detailed=True)

        return {
            "description": result.caption,
            "text_detected": result.text_detected,
            "metadata": result.metadata,
        }


# ============================================================================
# Video Handler
# ============================================================================


class VideoHandler(BaseInputHandler):
    """
    Handler for video input processing.

    Extracts frames and audio for analysis.

    Example:
        >>> handler = VideoHandler()
        >>> await handler.initialize()
        >>> result = await handler.extract_frames("video.mp4", max_frames=10)
        >>> print(f"Extracted {result.frame_count} frames")
    """

    def __init__(
        self,
        frame_extraction_mode: str = "uniform",  # uniform, keyframe, scene
        sample_rate: float = 1.0,  # frames per second for uniform
        max_frames: int = 100,
    ) -> None:
        """
        Initialize video handler.

        Args:
            frame_extraction_mode: How to extract frames
            sample_rate: Frames per second for uniform sampling
            max_frames: Maximum frames to extract
        """
        super().__init__()
        self._extraction_mode = frame_extraction_mode
        self._sample_rate = sample_rate
        self._max_frames = max_frames

    async def initialize(self) -> None:
        """Initialize video handler."""
        # Check for required dependencies
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python is required for video processing. "
                "Install it with: pip install opencv-python"
            )

        self._initialized = True
        logger.info("VideoHandler initialized")

    async def close(self) -> None:
        """Close video handler."""
        self._initialized = False

    def _validate_video_file(self, file_path: str) -> None:
        """Validate video file."""
        path = Path(file_path)

        if not path.exists():
            raise ValidationError(f"Video file not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix not in VIDEO_FORMATS:
            raise ValidationError(
                f"Unsupported video format: {suffix}. "
                f"Supported formats: {', '.join(VIDEO_FORMATS)}"
            )

    async def extract_frames(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        extract_audio: bool = False,
    ) -> VideoFrameResult:
        """
        Extract frames from video.

        Args:
            video_path: Path to video file
            max_frames: Override max frames (uses instance default if None)
            extract_audio: Also extract audio track

        Returns:
            VideoFrameResult with extracted frames
        """
        if not self._initialized:
            await self.initialize()

        self._validate_video_file(video_path)

        import cv2

        cap = cv2.VideoCapture(video_path)

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0

            effective_max = max_frames or self._max_frames
            frames: List[bytes] = []

            if self._extraction_mode == "uniform":
                # Extract frames at uniform intervals
                frame_interval = max(1, int(fps / self._sample_rate))
                frame_indices = list(range(0, total_frames, frame_interval))[:effective_max]
            else:
                # Default to uniform sampling
                frame_indices = list(range(0, total_frames, max(1, total_frames // effective_max)))[:effective_max]

            loop = asyncio.get_event_loop()

            def extract():
                extracted = []
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Encode frame as PNG bytes
                        _, buffer = cv2.imencode(".png", frame)
                        extracted.append(buffer.tobytes())
                return extracted

            frames = await loop.run_in_executor(None, extract)

            # Extract audio if requested
            audio_path = None
            if extract_audio:
                audio_path = await self._extract_audio(video_path)

            return VideoFrameResult(
                frames=frames,
                frame_count=len(frames),
                duration_seconds=duration,
                fps=fps,
                width=width,
                height=height,
                audio_path=audio_path,
                metadata={
                    "extraction_mode": self._extraction_mode,
                    "sample_rate": self._sample_rate,
                    "source": video_path,
                },
            )

        finally:
            cap.release()

    async def _extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio track from video."""
        try:
            import subprocess

            # Create temp file for audio
            temp_audio = tempfile.NamedTemporaryFile(
                delete=False, suffix=".wav"
            )
            temp_audio.close()

            # Use ffmpeg to extract audio
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "ffmpeg", "-i", video_path,
                        "-vn", "-acodec", "pcm_s16le",
                        "-ar", "16000", "-ac", "1",
                        "-y", temp_audio.name
                    ],
                    capture_output=True,
                    check=True,
                )
            )

            return temp_audio.name
        except Exception as e:
            logger.warning(f"Failed to extract audio: {e}")
            return None


# ============================================================================
# Factory Functions
# ============================================================================


def get_audio_handler(
    engine: str = "whisper",
    **kwargs: Any,
) -> AudioHandler:
    """
    Create an audio handler.

    Args:
        engine: Transcription engine (whisper, whisper_api, groq_whisper)
        **kwargs: Additional arguments for AudioHandler

    Returns:
        Configured AudioHandler instance
    """
    return AudioHandler(engine=engine, **kwargs)


def get_image_handler(
    model: str = "gpt4v",
    **kwargs: Any,
) -> ImageHandler:
    """
    Create an image handler.

    Args:
        model: Captioning model (gpt4v, claude_vision, gemini_vision, blip)
        **kwargs: Additional arguments for ImageHandler

    Returns:
        Configured ImageHandler instance
    """
    return ImageHandler(model=model, **kwargs)


def get_video_handler(**kwargs: Any) -> VideoHandler:
    """
    Create a video handler.

    Args:
        **kwargs: Additional arguments for VideoHandler

    Returns:
        Configured VideoHandler instance
    """
    return VideoHandler(**kwargs)
