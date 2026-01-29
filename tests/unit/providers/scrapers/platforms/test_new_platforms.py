"""
Unit tests for New Platform Scrapers.

Tests cover:
- IMDBScraper
- YelpScraper
- TrustpilotScraper
- GoogleReviewsScraper
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from sentimatrix.providers.base import ProviderType


# ============================================================================
# IMDBScraper Tests
# ============================================================================

class TestIMDBScraper:
    """Test IMDB scraper."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.platforms.imdb import IMDBScraper, IMDBConfig

        config = IMDBConfig(omdb_api_key="test_key")
        scraper = IMDBScraper(config)

        assert scraper._imdb_config.omdb_api_key == "test_key"
        assert scraper.platform_name == "imdb"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.platforms.imdb import IMDBScraper

        scraper = IMDBScraper()

        info = scraper.info
        assert info.name == "imdb"
        assert info.provider_type == ProviderType.SCRAPER
        assert "movie" in info.description.lower() or "IMDB" in info.description

    def test_validate_url(self):
        """Test URL validation."""
        from sentimatrix.providers.scrapers.platforms.imdb import IMDBScraper

        scraper = IMDBScraper()

        # Valid URLs
        assert scraper.validate_url("https://www.imdb.com/title/tt0111161/") is True
        assert scraper.validate_url("https://imdb.com/title/tt1234567/reviews") is True

        # Invalid URLs
        assert scraper.validate_url("https://google.com") is False
        assert scraper.validate_url("https://amazon.com/dp/B08N5WRWNW") is False

    def test_extract_id(self):
        """Test title ID extraction."""
        from sentimatrix.providers.scrapers.platforms.imdb import IMDBScraper

        scraper = IMDBScraper()

        # Valid extractions
        assert scraper.extract_id("https://www.imdb.com/title/tt0111161/") == "tt0111161"
        assert scraper.extract_id("https://imdb.com/title/tt1234567/reviews") == "tt1234567"

        # Invalid
        assert scraper.extract_id("https://google.com") is None

    def test_validate_title_id(self):
        """Test title ID format validation."""
        from sentimatrix.providers.scrapers.platforms.imdb import IMDBScraper

        scraper = IMDBScraper()

        # Valid IDs
        assert scraper.validate_title_id("tt0111161") is True
        assert scraper.validate_title_id("tt12345678") is True

        # Invalid IDs
        assert scraper.validate_title_id("0111161") is False
        assert scraper.validate_title_id("tt111") is False


# ============================================================================
# YelpScraper Tests
# ============================================================================

class TestYelpScraper:
    """Test Yelp scraper."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.platforms.yelp import YelpScraper, YelpConfig

        config = YelpConfig(api_key="test_key", location="San Francisco, CA")
        scraper = YelpScraper(config)

        assert scraper._yelp_config.api_key == "test_key"
        assert scraper._yelp_config.location == "San Francisco, CA"
        assert scraper.platform_name == "yelp"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.platforms.yelp import YelpScraper

        scraper = YelpScraper()

        info = scraper.info
        assert info.name == "yelp"
        assert info.provider_type == ProviderType.SCRAPER
        assert "business" in info.description.lower() or "Yelp" in info.description

    def test_validate_url(self):
        """Test URL validation."""
        from sentimatrix.providers.scrapers.platforms.yelp import YelpScraper

        scraper = YelpScraper()

        # Valid URLs
        assert scraper.validate_url("https://www.yelp.com/biz/the-french-laundry-yountville") is True
        assert scraper.validate_url("https://yelp.com/biz/some-restaurant-city") is True

        # Invalid URLs
        assert scraper.validate_url("https://google.com") is False
        assert scraper.validate_url("https://trustpilot.com/review/amazon.com") is False

    def test_extract_id(self):
        """Test business ID extraction."""
        from sentimatrix.providers.scrapers.platforms.yelp import YelpScraper

        scraper = YelpScraper()

        # Valid extractions
        assert scraper.extract_id("https://www.yelp.com/biz/the-french-laundry-yountville") == "the-french-laundry-yountville"
        assert scraper.extract_id("https://yelp.com/biz/some-business-id-123") == "some-business-id-123"

        # Invalid
        assert scraper.extract_id("https://google.com") is None


# ============================================================================
# TrustpilotScraper Tests
# ============================================================================

class TestTrustpilotScraper:
    """Test Trustpilot scraper."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.platforms.trustpilot import TrustpilotScraper, TrustpilotConfig

        config = TrustpilotConfig(country="uk")
        scraper = TrustpilotScraper(config)

        assert scraper._trustpilot_config.country == "uk"
        assert scraper.platform_name == "trustpilot"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.platforms.trustpilot import TrustpilotScraper

        scraper = TrustpilotScraper()

        info = scraper.info
        assert info.name == "trustpilot"
        assert info.provider_type == ProviderType.SCRAPER
        assert "company" in info.description.lower() or "Trustpilot" in info.description

    def test_validate_url(self):
        """Test URL validation."""
        from sentimatrix.providers.scrapers.platforms.trustpilot import TrustpilotScraper

        scraper = TrustpilotScraper()

        # Valid URLs
        assert scraper.validate_url("https://www.trustpilot.com/review/amazon.com") is True
        assert scraper.validate_url("https://trustpilot.com/review/example.com") is True

        # Invalid URLs
        assert scraper.validate_url("https://google.com") is False
        assert scraper.validate_url("https://yelp.com/biz/test") is False

    def test_extract_id(self):
        """Test company ID extraction."""
        from sentimatrix.providers.scrapers.platforms.trustpilot import TrustpilotScraper

        scraper = TrustpilotScraper()

        # Valid extractions
        assert scraper.extract_id("https://www.trustpilot.com/review/amazon.com") == "amazon.com"
        assert scraper.extract_id("https://trustpilot.com/review/example.co.uk") == "example.co.uk"

        # Invalid
        assert scraper.extract_id("https://google.com") is None


# ============================================================================
# GoogleReviewsScraper Tests
# ============================================================================

class TestGoogleReviewsScraper:
    """Test Google Reviews scraper."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.platforms.google_reviews import GoogleReviewsScraper, GoogleReviewsConfig

        config = GoogleReviewsConfig(api_key="test_key", location="London, UK")
        scraper = GoogleReviewsScraper(config)

        assert scraper._google_config.api_key == "test_key"
        assert scraper._google_config.location == "London, UK"
        assert scraper.platform_name == "google_reviews"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.platforms.google_reviews import GoogleReviewsScraper

        scraper = GoogleReviewsScraper()

        info = scraper.info
        assert info.name == "google_reviews"
        assert info.provider_type == ProviderType.SCRAPER
        assert "Places" in info.description or "Google" in info.description

    def test_validate_url(self):
        """Test URL validation."""
        from sentimatrix.providers.scrapers.platforms.google_reviews import GoogleReviewsScraper

        scraper = GoogleReviewsScraper()

        # Valid URLs
        assert scraper.validate_url("https://www.google.com/maps/place/test") is True
        assert scraper.validate_url("https://maps.google.com/place/test") is True

        # Invalid URLs
        assert scraper.validate_url("https://yelp.com/biz/test") is False
        assert scraper.validate_url("https://trustpilot.com") is False


# ============================================================================
# Registration Tests
# ============================================================================

class TestPlatformScraperImports:
    """Test that all platform scrapers are properly importable."""

    def test_all_scrapers_importable(self):
        """Test all scrapers can be imported."""
        from sentimatrix.providers.scrapers.platforms import (
            AmazonScraper,
            SteamScraper,
            YouTubeScraper,
            RedditScraper,
            IMDBScraper,
            YelpScraper,
            TrustpilotScraper,
            GoogleReviewsScraper,
        )

        assert AmazonScraper is not None
        assert SteamScraper is not None
        assert YouTubeScraper is not None
        assert RedditScraper is not None
        assert IMDBScraper is not None
        assert YelpScraper is not None
        assert TrustpilotScraper is not None
        assert GoogleReviewsScraper is not None

    def test_base_classes_importable(self):
        """Test base classes can be imported."""
        from sentimatrix.providers.scrapers.platforms import (
            BasePlatformScraper,
            PlatformConfig,
            ProductInfo,
            ReviewFilter,
            SortOrder,
        )

        assert BasePlatformScraper is not None
        assert PlatformConfig is not None
        assert ProductInfo is not None
        assert ReviewFilter is not None
        assert SortOrder is not None


# ============================================================================
# Config Tests
# ============================================================================

class TestConfigClasses:
    """Test configuration classes."""

    def test_imdb_config_defaults(self):
        """Test IMDB config defaults."""
        from sentimatrix.providers.scrapers.platforms.imdb import IMDBConfig

        config = IMDBConfig()
        assert config.omdb_api_key is None
        assert config.include_spoilers is False
        assert config.requests_per_second == 1.0

    def test_yelp_config_defaults(self):
        """Test Yelp config defaults."""
        from sentimatrix.providers.scrapers.platforms.yelp import YelpConfig

        config = YelpConfig()
        assert config.api_key is None
        assert config.location == "New York, NY"
        assert config.requests_per_second == 0.5

    def test_trustpilot_config_defaults(self):
        """Test Trustpilot config defaults."""
        from sentimatrix.providers.scrapers.platforms.trustpilot import TrustpilotConfig

        config = TrustpilotConfig()
        assert config.country == "www"
        assert config.include_replies is True
        assert config.requests_per_second == 0.5

    def test_google_reviews_config_defaults(self):
        """Test Google Reviews config defaults."""
        from sentimatrix.providers.scrapers.platforms.google_reviews import GoogleReviewsConfig

        config = GoogleReviewsConfig()
        assert config.api_key is None
        assert config.serpapi_key is None
        assert config.location == "New York, NY"
        assert config.language == "en"


# ============================================================================
# Data Classes Tests
# ============================================================================

class TestDataClasses:
    """Test data classes."""

    def test_movie_info(self):
        """Test MovieInfo dataclass."""
        from sentimatrix.providers.scrapers.platforms.imdb import MovieInfo

        movie = MovieInfo(
            id="tt0111161",
            title="The Shawshank Redemption",
            year="1994",
            type="movie",
            rating=9.3,
        )

        assert movie.id == "tt0111161"
        assert movie.title == "The Shawshank Redemption"
        assert movie.rating == 9.3

        # Test to_dict
        data = movie.to_dict()
        assert data["id"] == "tt0111161"
        assert data["title"] == "The Shawshank Redemption"

    def test_business_info(self):
        """Test BusinessInfo dataclass."""
        from sentimatrix.providers.scrapers.platforms.yelp import BusinessInfo

        business = BusinessInfo(
            id="test-business",
            name="Test Restaurant",
            url="https://yelp.com/biz/test-business",
            rating=4.5,
            review_count=100,
        )

        assert business.id == "test-business"
        assert business.name == "Test Restaurant"
        assert business.rating == 4.5

        # Test to_dict
        data = business.to_dict()
        assert data["id"] == "test-business"
        assert data["review_count"] == 100

    def test_company_info(self):
        """Test CompanyInfo dataclass."""
        from sentimatrix.providers.scrapers.platforms.trustpilot import CompanyInfo

        company = CompanyInfo(
            id="amazon.com",
            name="Amazon",
            url="https://trustpilot.com/review/amazon.com",
            rating=1.6,
            review_count=50000,
        )

        assert company.id == "amazon.com"
        assert company.name == "Amazon"
        assert company.rating == 1.6

        # Test to_dict
        data = company.to_dict()
        assert data["id"] == "amazon.com"
        assert data["review_count"] == 50000

    def test_place_info(self):
        """Test PlaceInfo dataclass."""
        from sentimatrix.providers.scrapers.platforms.google_reviews import PlaceInfo

        place = PlaceInfo(
            place_id="ChIJN1t_tDeuEmsRUsoyG83frY4",
            name="Google Sydney",
            formatted_address="48 Pirrama Rd, Pyrmont NSW 2009, Australia",
            rating=4.4,
            user_ratings_total=1000,
        )

        assert place.place_id == "ChIJN1t_tDeuEmsRUsoyG83frY4"
        assert place.name == "Google Sydney"
        assert place.rating == 4.4

        # Test to_dict
        data = place.to_dict()
        assert data["place_id"] == "ChIJN1t_tDeuEmsRUsoyG83frY4"
        assert data["user_ratings_total"] == 1000
