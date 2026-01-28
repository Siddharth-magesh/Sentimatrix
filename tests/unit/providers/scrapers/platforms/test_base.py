"""
Unit tests for Base Platform Scraper.

Tests cover:
- PlatformConfig and ProductInfo dataclasses
- URL validation helpers
- Text cleaning and date parsing utilities
- Rating normalization
- Review ID generation
"""

from datetime import datetime

import pytest

from sentimatrix.providers.scrapers.platforms.base import (
    BasePlatformScraper,
    PlatformConfig,
    ProductInfo,
    SortOrder,
    ReviewFilter,
)


class TestPlatformConfig:
    """Test PlatformConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PlatformConfig()

        assert config.requests_per_second == 1.0
        assert config.burst_size == 5
        assert config.max_retries == 3
        assert config.timeout == 30
        assert config.headless is True
        assert config.country == "us"
        assert config.language == "en"

    def test_custom_config(self):
        """Test custom configuration."""
        config = PlatformConfig(
            requests_per_second=0.5,
            max_retries=5,
            country="uk",
            api_key="test_key",
        )

        assert config.requests_per_second == 0.5
        assert config.max_retries == 5
        assert config.country == "uk"
        assert config.api_key == "test_key"

    def test_to_scraper_config(self):
        """Test conversion to ScraperConfig."""
        config = PlatformConfig(timeout=60, headless=False)
        scraper_config = config.to_scraper_config()

        assert scraper_config.timeout == 60
        assert scraper_config.headless is False


class TestProductInfo:
    """Test ProductInfo dataclass."""

    def test_basic_creation(self):
        """Test basic product info creation."""
        info = ProductInfo(
            id="123",
            name="Test Product",
            platform="amazon",
        )

        assert info.id == "123"
        assert info.name == "Test Product"
        assert info.platform == "amazon"
        assert info.url is None
        assert info.rating is None

    def test_full_creation(self):
        """Test product info with all fields."""
        info = ProductInfo(
            id="456",
            name="Full Product",
            platform="steam",
            url="https://store.steampowered.com/app/456",
            description="A great game",
            price=29.99,
            currency="USD",
            rating=4.5,
            review_count=1000,
            image_url="https://example.com/image.jpg",
            category="Games",
            metadata={"developer": "Test Studio"},
        )

        assert info.price == 29.99
        assert info.rating == 4.5
        assert info.metadata["developer"] == "Test Studio"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        info = ProductInfo(
            id="789",
            name="Dict Product",
            platform="youtube",
            rating=4.0,
        )

        data = info.to_dict()

        assert data["id"] == "789"
        assert data["name"] == "Dict Product"
        assert data["platform"] == "youtube"
        assert data["rating"] == 4.0


class TestSortOrder:
    """Test SortOrder enum."""

    def test_sort_values(self):
        """Test sort order values."""
        assert SortOrder.RECENT.value == "recent"
        assert SortOrder.HELPFUL.value == "helpful"
        assert SortOrder.RATING_HIGH.value == "rating_high"
        assert SortOrder.RATING_LOW.value == "rating_low"
        assert SortOrder.RELEVANCE.value == "relevance"


class TestReviewFilter:
    """Test ReviewFilter enum."""

    def test_filter_values(self):
        """Test filter values."""
        assert ReviewFilter.ALL.value == "all"
        assert ReviewFilter.POSITIVE.value == "positive"
        assert ReviewFilter.NEGATIVE.value == "negative"
        assert ReviewFilter.VERIFIED.value == "verified"


class TestBasePlatformScraperHelpers:
    """Test helper methods on BasePlatformScraper."""

    def test_clean_text(self):
        """Test text cleaning."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper.__new__(AmazonScraper)
        scraper._platform_config = PlatformConfig()

        # Test whitespace normalization
        assert scraper.clean_text("  hello   world  ") == "hello world"
        assert scraper.clean_text("line1\n\nline2") == "line1 line2"
        assert scraper.clean_text("") == ""

    def test_normalize_rating(self):
        """Test rating normalization."""
        from sentimatrix.providers.scrapers.platforms.steam import SteamScraper

        scraper = SteamScraper.__new__(SteamScraper)
        scraper._platform_config = PlatformConfig()

        # 0-100 to 0-5
        assert scraper.normalize_rating(100, 0, 100, 5.0) == 5.0
        assert scraper.normalize_rating(50, 0, 100, 5.0) == 2.5
        assert scraper.normalize_rating(0, 0, 100, 5.0) == 0.0

        # 1-10 to 0-5
        assert scraper.normalize_rating(10, 1, 10, 5.0) == 5.0
        assert scraper.normalize_rating(5.5, 1, 10, 5.0) == 2.5

    def test_parse_date(self):
        """Test date parsing."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper.__new__(AmazonScraper)
        scraper._platform_config = PlatformConfig()

        # ISO format
        dt = scraper.parse_date("2024-01-15")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 15

        # Long format
        dt = scraper.parse_date("January 15, 2024")
        assert dt.year == 2024
        assert dt.month == 1

        # Invalid format
        dt = scraper.parse_date("not a date")
        assert dt is None

        # Empty string
        dt = scraper.parse_date("")
        assert dt is None

    def test_generate_review_id(self):
        """Test review ID generation."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper.__new__(AmazonScraper)
        scraper._platform_config = PlatformConfig()

        # Same input should generate same ID
        id1 = scraper.generate_review_id("amazon", "Great product!", "user1")
        id2 = scraper.generate_review_id("amazon", "Great product!", "user1")
        assert id1 == id2

        # Different input should generate different ID
        id3 = scraper.generate_review_id("amazon", "Bad product!", "user1")
        assert id1 != id3

        # ID should be 16 characters (hex)
        assert len(id1) == 16
