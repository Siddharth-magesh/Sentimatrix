"""
Unit tests for Sentimatrix Input Handlers.

Tests:
- AudioHandler: Audio transcription
- ImageHandler: Image captioning
- VideoHandler: Video frame extraction
"""

import asyncio
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    AudioHandler,
    ImageHandler,
    VideoHandler,
    # Factory functions
    get_audio_handler,
    get_image_handler,
    get_video_handler,
)


# ============================================================================
# Test Data Classes
# ============================================================================


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_transcription_result_creation(self):
        """Test basic TranscriptionResult creation."""
        result = TranscriptionResult(text="Hello world")
        assert result.text == "Hello world"
        assert result.language is None
        assert result.segments == []

    def test_transcription_result_full(self):
        """Test TranscriptionResult with all fields."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration_seconds=5.0,
            segments=[{"start": 0, "end": 2, "text": "Hello"}],
            confidence=0.95,
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration_seconds == 5.0
        assert len(result.segments) == 1
        assert result.confidence == 0.95

    def test_transcription_result_to_dict(self):
        """Test TranscriptionResult to_dict method."""
        result = TranscriptionResult(
            text="Hello",
            language="en",
            duration_seconds=1.0,
        )
        d = result.to_dict()
        assert d["text"] == "Hello"
        assert d["language"] == "en"
        assert d["duration_seconds"] == 1.0


class TestCaptionResult:
    """Tests for CaptionResult dataclass."""

    def test_caption_result_creation(self):
        """Test basic CaptionResult creation."""
        result = CaptionResult(caption="A beautiful sunset")
        assert result.caption == "A beautiful sunset"
        assert result.confidence is None
        assert result.objects is None

    def test_caption_result_full(self):
        """Test CaptionResult with all fields."""
        result = CaptionResult(
            caption="A dog playing in the park",
            confidence=0.92,
            objects=["dog", "park", "grass"],
            text_detected="Welcome",
        )
        assert result.caption == "A dog playing in the park"
        assert result.confidence == 0.92
        assert "dog" in result.objects
        assert result.text_detected == "Welcome"

    def test_caption_result_to_dict(self):
        """Test CaptionResult to_dict method."""
        result = CaptionResult(
            caption="A test image",
            confidence=0.85,
        )
        d = result.to_dict()
        assert d["caption"] == "A test image"
        assert d["confidence"] == 0.85


class TestVideoFrameResult:
    """Tests for VideoFrameResult dataclass."""

    def test_video_frame_result_creation(self):
        """Test basic VideoFrameResult creation."""
        result = VideoFrameResult(
            frames=[b"frame1", b"frame2"],
            frame_count=2,
            duration_seconds=10.0,
            fps=30.0,
            width=1920,
            height=1080,
        )
        assert len(result.frames) == 2
        assert result.frame_count == 2
        assert result.duration_seconds == 10.0
        assert result.fps == 30.0
        assert result.width == 1920
        assert result.height == 1080

    def test_video_frame_result_to_dict(self):
        """Test VideoFrameResult to_dict method."""
        result = VideoFrameResult(
            frames=[],
            frame_count=5,
            duration_seconds=15.0,
            fps=24.0,
            width=1280,
            height=720,
        )
        d = result.to_dict()
        assert d["frame_count"] == 5
        assert d["duration_seconds"] == 15.0
        assert d["fps"] == 24.0
        assert "frames" not in d  # frames excluded from dict


# ============================================================================
# Audio Handler Tests
# ============================================================================


class TestAudioHandler:
    """Tests for AudioHandler."""

    def test_audio_handler_init(self):
        """Test AudioHandler initialization."""
        handler = AudioHandler(
            engine=AudioEngine.WHISPER_LOCAL,
            model="base",
            language="en",
        )
        assert handler._engine == AudioEngine.WHISPER_LOCAL
        assert handler._model_name == "base"
        assert handler._language == "en"
        assert handler._initialized is False

    def test_audio_handler_engine_types(self):
        """Test different engine types."""
        for engine in AudioEngine:
            handler = AudioHandler(engine=engine)
            assert handler._engine == engine

    def test_audio_handler_string_engine(self):
        """Test AudioHandler with string engine."""
        handler = AudioHandler(engine="whisper")
        assert handler._engine == AudioEngine.WHISPER_LOCAL

    @pytest.mark.asyncio
    async def test_audio_handler_validate_file_missing(self, tmp_path):
        """Test validation of missing audio file."""
        handler = AudioHandler()
        with pytest.raises(Exception) as excinfo:
            handler._validate_audio_file(str(tmp_path / "nonexistent.wav"))
        assert "not found" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_audio_handler_validate_file_wrong_format(self, tmp_path):
        """Test validation of wrong audio format."""
        wrong_file = tmp_path / "test.txt"
        wrong_file.write_text("not audio")

        handler = AudioHandler()
        with pytest.raises(Exception) as excinfo:
            handler._validate_audio_file(str(wrong_file))
        assert "unsupported" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_audio_handler_validate_file_valid(self, tmp_path):
        """Test validation of valid audio file."""
        valid_file = tmp_path / "test.wav"
        valid_file.write_bytes(b"RIFF....WAVEfmt ")

        handler = AudioHandler()
        # Should not raise
        handler._validate_audio_file(str(valid_file))

    @pytest.mark.asyncio
    async def test_audio_handler_context_manager(self):
        """Test AudioHandler context manager."""
        handler = AudioHandler(engine=AudioEngine.WHISPER_LOCAL)

        # Mock the initialization
        handler.initialize = AsyncMock()
        handler.close = AsyncMock()

        async with handler:
            handler.initialize.assert_called_once()

        handler.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_audio_handler_transcribe_mock(self):
        """Test transcription with mocked whisper."""
        # Skip if whisper is not installed
        try:
            import whisper
        except ImportError:
            pytest.skip("openai-whisper not installed")

        handler = AudioHandler(engine=AudioEngine.WHISPER_LOCAL)
        handler._initialized = True

        # Create a mock model
        mock_model = MagicMock()
        mock_model.transcribe = MagicMock(return_value={
            "text": " Hello, world!",
            "language": "en",
            "segments": [],
        })
        handler._model = mock_model

        # Create a temp audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(b"RIFF....WAVEfmt ")
            temp_path = f.name

        try:
            result = await handler._transcribe_whisper_local(
                temp_path, None, False, False
            )
            assert result.text == "Hello, world!"
            assert result.language == "en"
        finally:
            os.unlink(temp_path)


# ============================================================================
# Image Handler Tests
# ============================================================================


class TestImageHandler:
    """Tests for ImageHandler."""

    def test_image_handler_init(self):
        """Test ImageHandler initialization."""
        handler = ImageHandler(
            model=ImageModel.GPT4V,
            api_key="test_key",
            include_ocr=True,
        )
        assert handler._model_type == ImageModel.GPT4V
        assert handler._api_key == "test_key"
        assert handler._include_ocr is True
        assert handler._initialized is False

    def test_image_handler_model_types(self):
        """Test different model types."""
        for model in ImageModel:
            handler = ImageHandler(model=model)
            assert handler._model_type == model

    def test_image_handler_string_model(self):
        """Test ImageHandler with string model."""
        handler = ImageHandler(model="gpt4v")
        assert handler._model_type == ImageModel.GPT4V

    @pytest.mark.asyncio
    async def test_image_handler_validate_file_missing(self, tmp_path):
        """Test validation of missing image file."""
        handler = ImageHandler()
        with pytest.raises(Exception) as excinfo:
            handler._validate_image_file(str(tmp_path / "nonexistent.png"))
        assert "not found" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_image_handler_validate_file_wrong_format(self, tmp_path):
        """Test validation of wrong image format."""
        wrong_file = tmp_path / "test.txt"
        wrong_file.write_text("not image")

        handler = ImageHandler()
        with pytest.raises(Exception) as excinfo:
            handler._validate_image_file(str(wrong_file))
        assert "unsupported" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_image_handler_validate_file_valid(self, tmp_path):
        """Test validation of valid image file."""
        valid_file = tmp_path / "test.png"
        # Write minimal PNG header
        valid_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        handler = ImageHandler()
        # Should not raise
        handler._validate_image_file(str(valid_file))

    def test_image_handler_encode_base64(self):
        """Test base64 encoding."""
        handler = ImageHandler()
        test_bytes = b"test image data"
        encoded = handler._encode_image_base64(test_bytes)
        assert isinstance(encoded, str)
        # Verify it's valid base64
        import base64
        decoded = base64.b64decode(encoded)
        assert decoded == test_bytes

    @pytest.mark.asyncio
    async def test_image_handler_context_manager(self):
        """Test ImageHandler context manager."""
        handler = ImageHandler(model=ImageModel.GPT4V)

        # Mock the initialization
        handler.initialize = AsyncMock()
        handler.close = AsyncMock()

        async with handler:
            handler.initialize.assert_called_once()

        handler.close.assert_called_once()


# ============================================================================
# Video Handler Tests
# ============================================================================


class TestVideoHandler:
    """Tests for VideoHandler."""

    def test_video_handler_init(self):
        """Test VideoHandler initialization."""
        handler = VideoHandler(
            frame_extraction_mode="uniform",
            sample_rate=1.0,
            max_frames=50,
        )
        assert handler._extraction_mode == "uniform"
        assert handler._sample_rate == 1.0
        assert handler._max_frames == 50
        assert handler._initialized is False

    @pytest.mark.asyncio
    async def test_video_handler_validate_file_missing(self, tmp_path):
        """Test validation of missing video file."""
        handler = VideoHandler()
        with pytest.raises(Exception) as excinfo:
            handler._validate_video_file(str(tmp_path / "nonexistent.mp4"))
        assert "not found" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_video_handler_validate_file_wrong_format(self, tmp_path):
        """Test validation of wrong video format."""
        wrong_file = tmp_path / "test.txt"
        wrong_file.write_text("not video")

        handler = VideoHandler()
        with pytest.raises(Exception) as excinfo:
            handler._validate_video_file(str(wrong_file))
        assert "unsupported" in str(excinfo.value).lower()

    @pytest.mark.asyncio
    async def test_video_handler_validate_file_valid(self, tmp_path):
        """Test validation of valid video file."""
        valid_file = tmp_path / "test.mp4"
        valid_file.write_bytes(b"\x00\x00\x00\x20ftypisom")

        handler = VideoHandler()
        # Should not raise
        handler._validate_video_file(str(valid_file))

    @pytest.mark.asyncio
    async def test_video_handler_context_manager(self):
        """Test VideoHandler context manager."""
        handler = VideoHandler()

        # Mock the initialization
        handler.initialize = AsyncMock()
        handler.close = AsyncMock()

        async with handler:
            handler.initialize.assert_called_once()

        handler.close.assert_called_once()


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_audio_handler(self):
        """Test get_audio_handler factory."""
        handler = get_audio_handler(engine="whisper", model="base")
        assert isinstance(handler, AudioHandler)
        assert handler._engine == AudioEngine.WHISPER_LOCAL

    def test_get_image_handler(self):
        """Test get_image_handler factory."""
        handler = get_image_handler(model="gpt4v")
        assert isinstance(handler, ImageHandler)
        assert handler._model_type == ImageModel.GPT4V

    def test_get_video_handler(self):
        """Test get_video_handler factory."""
        handler = get_video_handler(max_frames=50)
        assert isinstance(handler, VideoHandler)
        assert handler._max_frames == 50


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Tests for format constants."""

    def test_audio_formats(self):
        """Test AUDIO_FORMATS constant."""
        assert ".wav" in AUDIO_FORMATS
        assert ".mp3" in AUDIO_FORMATS
        assert ".flac" in AUDIO_FORMATS
        assert ".txt" not in AUDIO_FORMATS

    def test_image_formats(self):
        """Test IMAGE_FORMATS constant."""
        assert ".png" in IMAGE_FORMATS
        assert ".jpg" in IMAGE_FORMATS
        assert ".jpeg" in IMAGE_FORMATS
        assert ".txt" not in IMAGE_FORMATS

    def test_video_formats(self):
        """Test VIDEO_FORMATS constant."""
        assert ".mp4" in VIDEO_FORMATS
        assert ".avi" in VIDEO_FORMATS
        assert ".mov" in VIDEO_FORMATS
        assert ".txt" not in VIDEO_FORMATS


# ============================================================================
# Enum Tests
# ============================================================================


class TestEnums:
    """Tests for enums."""

    def test_audio_engine_values(self):
        """Test AudioEngine enum values."""
        assert AudioEngine.WHISPER_LOCAL.value == "whisper"
        assert AudioEngine.WHISPER_API.value == "whisper_api"
        assert AudioEngine.GROQ_WHISPER.value == "groq_whisper"

    def test_image_model_values(self):
        """Test ImageModel enum values."""
        assert ImageModel.GPT4V.value == "gpt4v"
        assert ImageModel.CLAUDE_VISION.value == "claude_vision"
        assert ImageModel.GEMINI_VISION.value == "gemini_vision"
        assert ImageModel.BLIP.value == "blip"
