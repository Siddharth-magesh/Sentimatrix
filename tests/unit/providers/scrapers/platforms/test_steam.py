"""
Unit tests for Steam Scraper.

Tests cover:
- App ID validation and extraction
- URL validation
- API response parsing
- Configuration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from sentimatrix.providers.base import ProviderType


class TestSteamConfig:
    """Test SteamConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamConfig

        config = SteamConfig()

        assert config.language == "english"
        assert config.purchase_type == "all"
        assert config.review_type == "all"
        assert config.day_range == 0
        assert config.requests_per_second == 2.0

    def test_custom_config(self):
        """Test custom configuration."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamConfig

        config = SteamConfig(
            language="german",
            review_type="positive",
            day_range=30,
        )

        assert config.language == "german"
        assert config.review_type == "positive"
        assert config.day_range == 30


class TestSteamScraperInit:
    """Test Steam scraper initialization."""

    def test_init_default(self):
        """Test default initialization."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        assert scraper.platform_name == "steam"
        assert scraper.platform_domain == "store.steampowered.com"
        assert not scraper._initialized

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()
        info = scraper.info

        assert info.name == "steam"
        assert info.provider_type == ProviderType.SCRAPER
        assert info.capabilities.javascript_rendering is False


class TestSteamScraperValidation:
    """Test URL and app ID validation."""

    def test_validate_app_id_valid(self):
        """Test valid app ID validation."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        assert scraper.validate_app_id("730") is True
        assert scraper.validate_app_id("570") is True
        assert scraper.validate_app_id("1234567") is True

    def test_validate_app_id_invalid(self):
        """Test invalid app ID validation."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        assert scraper.validate_app_id("abc") is False
        assert scraper.validate_app_id("12.34") is False

    def test_validate_url_valid(self):
        """Test valid URL validation."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        assert scraper.validate_url("https://store.steampowered.com/app/730") is True
        assert scraper.validate_url("https://steamcommunity.com/app/570") is True

    def test_validate_url_invalid(self):
        """Test invalid URL validation."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        assert scraper.validate_url("https://example.com") is False
        assert scraper.validate_url("https://store.steampowered.com/search") is False

    def test_extract_id_from_store_url(self):
        """Test app ID extraction from store URL."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        app_id = scraper.extract_id("https://store.steampowered.com/app/730/CounterStrike_2/")
        assert app_id == "730"

    def test_extract_id_from_community_url(self):
        """Test app ID extraction from community URL."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        app_id = scraper.extract_id("https://steamcommunity.com/app/570")
        assert app_id == "570"


class TestSteamScraperParsing:
    """Test API response parsing."""

    def test_parse_review_positive(self):
        """Test parsing a positive review."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        review_data = {
            "recommendationid": "12345",
            "review": "Great game, highly recommend!",
            "voted_up": True,
            "author": {
                "steamid": "76561198000000000",
                "playtime_forever": 1200,
                "playtime_at_review": 600,
            },
            "timestamp_created": 1704067200,  # 2024-01-01
            "votes_up": 50,
            "votes_funny": 5,
            "steam_purchase": True,
            "received_for_free": False,
            "written_during_early_access": False,
            "language": "english",
            "weighted_vote_score": 0.85,
        }

        review = scraper._parse_review(review_data, "730")

        assert review is not None
        assert review.id == "12345"
        assert "Great game" in review.text
        assert review.rating == 5.0  # Positive = 5.0
        assert review.metadata["voted_up"] is True
        assert review.metadata["playtime_forever_hours"] == 20.0
        assert review.metadata["votes_up"] == 50

    def test_parse_review_negative(self):
        """Test parsing a negative review."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        review_data = {
            "recommendationid": "67890",
            "review": "Too many bugs.",
            "voted_up": False,
            "author": {"steamid": "123", "playtime_forever": 60},
            "timestamp_created": 1704067200,
        }

        review = scraper._parse_review(review_data, "730")

        assert review is not None
        assert review.rating == 1.0  # Negative = 1.0
        assert review.metadata["voted_up"] is False

    def test_parse_review_empty(self):
        """Test parsing review with no text returns None."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        review_data = {
            "review": "",
            "voted_up": True,
        }

        review = scraper._parse_review(review_data, "730")

        assert review is None


class TestSteamScraperScrape:
    """Test scraping functionality."""

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self):
        """Test scraping without initialization raises error."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper
        from sentimatrix.core.exceptions import ProviderInitializationError

        scraper = SteamScraper()

        with pytest.raises(ProviderInitializationError):
            await scraper.scrape_reviews("730")

    @pytest.mark.asyncio
    async def test_scrape_invalid_app_id(self):
        """Test scraping with invalid app ID."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()
        scraper._initialized = True

        with pytest.raises(ValueError, match="Invalid app ID"):
            await scraper.scrape_reviews("invalid")


class TestSteamScraperGameInfo:
    """Test game info parsing."""

    def test_parse_game_details(self):
        """Test parsing game details from API."""
        # This would need the full _fetch_json mock
        # Simplified test for now
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()

        # Verify scraper has the method
        assert hasattr(scraper, "get_product_info")
        assert hasattr(scraper, "search_games")
        assert hasattr(scraper, "get_review_summary")


class TestSteamScraperClose:
    """Test scraper close method."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing scraper."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper()
        scraper._initialized = True
        scraper._httpx_scraper = AsyncMock()

        await scraper.close()

        assert not scraper._initialized
        assert scraper._httpx_scraper is None
