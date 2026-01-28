"""
Unit tests for YouTube Scraper.

Tests cover:
- Video ID validation and extraction
- URL validation
- Comment parsing
- Configuration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sentimatrix.providers.base import ProviderType


class TestYouTubeConfig:
    """Test YouTubeConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeConfig

        config = YouTubeConfig()

        assert config.api_key is None
        assert config.include_replies is True
        assert config.max_reply_depth == 1
        assert config.requests_per_second == 1.0
        assert config.transcript_languages == ["en", "en-US"]

    def test_custom_config(self):
        """Test custom configuration."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeConfig

        config = YouTubeConfig(
            api_key="test_api_key",
            include_replies=False,
            transcript_languages=["de", "en"],
        )

        assert config.api_key == "test_api_key"
        assert config.include_replies is False
        assert config.transcript_languages == ["de", "en"]


class TestYouTubeScraperInit:
    """Test YouTube scraper initialization."""

    def test_init_default(self):
        """Test default initialization."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        assert scraper.platform_name == "youtube"
        assert scraper.platform_domain == "youtube.com"
        assert not scraper._initialized

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()
        info = scraper.info

        assert info.name == "youtube"
        assert info.provider_type == ProviderType.SCRAPER
        assert info.capabilities.javascript_rendering is False


class TestYouTubeScraperValidation:
    """Test URL and video ID validation."""

    def test_validate_video_id_valid(self):
        """Test valid video ID validation."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        assert scraper.validate_video_id("dQw4w9WgXcQ") is True
        assert scraper.validate_video_id("abc123_-XYZ") is True
        assert scraper.validate_video_id("12345678901") is True

    def test_validate_video_id_invalid(self):
        """Test invalid video ID validation."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        assert scraper.validate_video_id("short") is False
        assert scraper.validate_video_id("waytoolongid1234") is False
        assert scraper.validate_video_id("invalid!@#$") is False

    def test_validate_url_watch(self):
        """Test watch URL validation."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        assert scraper.validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
        assert scraper.validate_url("https://youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_validate_url_short(self):
        """Test short URL validation."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        assert scraper.validate_url("https://youtu.be/dQw4w9WgXcQ") is True

    def test_validate_url_embed(self):
        """Test embed URL validation."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        assert scraper.validate_url("https://youtube.com/embed/dQw4w9WgXcQ") is True

    def test_validate_url_shorts(self):
        """Test shorts URL validation."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        assert scraper.validate_url("https://youtube.com/shorts/dQw4w9WgXcQ") is True

    def test_validate_url_invalid(self):
        """Test invalid URL validation."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        assert scraper.validate_url("https://example.com") is False
        assert scraper.validate_url("https://vimeo.com/123") is False

    def test_extract_id_from_watch(self):
        """Test video ID extraction from watch URL."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        video_id = scraper.extract_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_id_from_short_url(self):
        """Test video ID extraction from short URL."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        video_id = scraper.extract_id("https://youtu.be/dQw4w9WgXcQ")
        assert video_id == "dQw4w9WgXcQ"

    def test_extract_id_with_params(self):
        """Test video ID extraction with additional params."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        video_id = scraper.extract_id("https://youtube.com/watch?v=dQw4w9WgXcQ&t=30s")
        assert video_id == "dQw4w9WgXcQ"


class TestYouTubeScraperParsing:
    """Test comment parsing."""

    def test_parse_comment_snippet(self):
        """Test parsing a comment snippet."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        snippet = {
            "textDisplay": "Great video!",
            "authorDisplayName": "TestUser",
            "authorChannelId": {"value": "UC123"},
            "publishedAt": "2024-01-15T10:00:00Z",
            "likeCount": 100,
        }

        comment = scraper._parse_comment_snippet(
            snippet, "dQw4w9WgXcQ", "comment123"
        )

        assert comment is not None
        assert comment.text == "Great video!"
        assert comment.author == "TestUser"
        assert comment.metadata["like_count"] == 100
        assert comment.metadata["video_id"] == "dQw4w9WgXcQ"

    def test_parse_comment_empty_text(self):
        """Test parsing comment with empty text returns None."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()

        snippet = {"textDisplay": ""}

        comment = scraper._parse_comment_snippet(
            snippet, "dQw4w9WgXcQ", "comment123"
        )

        assert comment is None


class TestYouTubeScraperVideoInfo:
    """Test VideoInfo dataclass."""

    def test_video_info_creation(self):
        """Test VideoInfo creation."""
        from sentimatrix.providers.scrapers.platforms.youtube import VideoInfo
        from datetime import datetime

        info = VideoInfo(
            id="dQw4w9WgXcQ",
            title="Test Video",
            channel_id="UC123",
            channel_title="Test Channel",
            description="Test description",
            view_count=1000000,
            like_count=50000,
            comment_count=5000,
        )

        assert info.id == "dQw4w9WgXcQ"
        assert info.view_count == 1000000
        assert info.tags == []

    def test_video_info_to_dict(self):
        """Test VideoInfo to_dict."""
        from sentimatrix.providers.scrapers.platforms.youtube import VideoInfo

        info = VideoInfo(
            id="abc123",
            title="Test",
            channel_id="ch1",
            channel_title="Channel",
            description="Desc",
        )

        data = info.to_dict()

        assert data["id"] == "abc123"
        assert data["title"] == "Test"


class TestYouTubeScraperTranscript:
    """Test Transcript dataclass."""

    def test_transcript_creation(self):
        """Test Transcript creation."""
        from sentimatrix.providers.scrapers.platforms.youtube import Transcript

        transcript = Transcript(
            video_id="dQw4w9WgXcQ",
            language="en",
            is_generated=True,
            segments=[
                {"text": "Hello", "start": 0.0, "duration": 1.0},
                {"text": "World", "start": 1.0, "duration": 1.0},
            ],
            full_text="Hello World",
        )

        assert transcript.video_id == "dQw4w9WgXcQ"
        assert transcript.is_generated is True
        assert len(transcript.segments) == 2

    def test_transcript_to_dict(self):
        """Test Transcript to_dict."""
        from sentimatrix.providers.scrapers.platforms.youtube import Transcript

        transcript = Transcript(
            video_id="abc",
            language="en",
            is_generated=False,
            segments=[],
            full_text="Test",
        )

        data = transcript.to_dict()

        assert data["video_id"] == "abc"
        assert data["full_text"] == "Test"


class TestYouTubeScraperScrape:
    """Test scraping functionality."""

    @pytest.mark.asyncio
    async def test_scrape_no_api_key(self):
        """Test scraping without API key raises error."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper
        from sentimatrix.core.exceptions import ProviderInitializationError

        scraper = YouTubeScraper()
        scraper._initialized = True

        with pytest.raises(ProviderInitializationError, match="API key"):
            await scraper.scrape_reviews("dQw4w9WgXcQ")

    @pytest.mark.asyncio
    async def test_scrape_invalid_video_id(self):
        """Test scraping with invalid video ID."""
        from sentimatrix.providers.scrapers.platforms.youtube import (
            YouTubeScraper,
            YouTubeConfig,
        )

        config = YouTubeConfig(api_key="test_key")
        scraper = YouTubeScraper(config)
        scraper._initialized = True

        with pytest.raises(ValueError, match="Invalid video ID"):
            await scraper.scrape_reviews("invalid")


class TestYouTubeScraperClose:
    """Test scraper close method."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing scraper."""
        from sentimatrix.providers.scrapers.platforms.youtube import YouTubeScraper

        scraper = YouTubeScraper()
        scraper._initialized = True
        scraper._httpx_scraper = AsyncMock()

        await scraper.close()

        assert not scraper._initialized
        assert scraper._httpx_scraper is None
