"""
Unit tests for Amazon Scraper.

Tests cover:
- ASIN validation and extraction
- URL validation
- Review page URL building
- HTML parsing (mocked)
- Configuration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sentimatrix.providers.base import ProviderType


class TestAmazonConfig:
    """Test AmazonConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonConfig

        config = AmazonConfig()

        assert config.country == "us"
        assert config.domain == "amazon.com"
        assert config.filter_verified is False
        assert config.requests_per_second == 0.5

    def test_uk_config(self):
        """Test UK configuration."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonConfig

        config = AmazonConfig(country="uk")

        assert config.domain == "amazon.co.uk"

    def test_custom_rate_limiting(self):
        """Test custom rate limiting."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonConfig

        config = AmazonConfig(
            requests_per_second=0.25,
            burst_size=2,
        )

        assert config.requests_per_second == 0.25
        assert config.burst_size == 2


class TestAmazonScraperInit:
    """Test Amazon scraper initialization."""

    def test_init_default(self):
        """Test default initialization."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        assert scraper.platform_name == "amazon"
        assert scraper.platform_domain == "amazon.com"
        assert not scraper._initialized

    def test_init_with_config(self):
        """Test initialization with config."""
        from sentimatrix.providers.scrapers.platforms.amazon import (
            AmazonScraper,
            AmazonConfig,
        )

        config = AmazonConfig(country="de")
        scraper = AmazonScraper(config)

        assert scraper.platform_domain == "amazon.de"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()
        info = scraper.info

        assert info.name == "amazon"
        assert info.provider_type == ProviderType.SCRAPER
        assert info.capabilities.javascript_rendering is True


class TestAmazonScraperValidation:
    """Test URL and ASIN validation."""

    def test_validate_asin_valid(self):
        """Test valid ASIN validation."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        assert scraper.validate_asin("B08N5WRWNW") is True
        assert scraper.validate_asin("0123456789") is True
        assert scraper.validate_asin("ABCDEFGHIJ") is True

    def test_validate_asin_invalid(self):
        """Test invalid ASIN validation."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        assert scraper.validate_asin("B08N5WRW") is False  # Too short
        assert scraper.validate_asin("B08N5WRWNWX") is False  # Too long
        assert scraper.validate_asin("B08N5WRW!W") is False  # Invalid char

    def test_validate_url_valid(self):
        """Test valid URL validation."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        assert scraper.validate_url("https://www.amazon.com/dp/B08N5WRWNW") is True
        assert scraper.validate_url("https://amazon.co.uk/gp/product/B08N5WRWNW") is True
        assert scraper.validate_url("https://www.amazon.de/dp/B08N5WRWNW") is True

    def test_validate_url_invalid(self):
        """Test invalid URL validation."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        assert scraper.validate_url("https://example.com") is False
        assert scraper.validate_url("https://ebay.com/item/123") is False
        assert scraper.validate_url("not a url") is False

    def test_extract_id_from_dp_url(self):
        """Test ASIN extraction from /dp/ URL."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        asin = scraper.extract_id("https://www.amazon.com/dp/B08N5WRWNW")
        assert asin == "B08N5WRWNW"

    def test_extract_id_from_product_url(self):
        """Test ASIN extraction from /gp/product/ URL."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        asin = scraper.extract_id("https://amazon.com/gp/product/B08N5WRWNW/ref=123")
        assert asin == "B08N5WRWNW"

    def test_extract_id_from_reviews_url(self):
        """Test ASIN extraction from reviews URL."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        asin = scraper.extract_id("https://amazon.com/product-reviews/B08N5WRWNW")
        assert asin == "B08N5WRWNW"

    def test_extract_id_invalid(self):
        """Test ASIN extraction from invalid URL."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        asin = scraper.extract_id("https://amazon.com/search?q=phone")
        assert asin is None


class TestAmazonScraperUrlBuilding:
    """Test URL building for reviews."""

    def test_build_reviews_url_basic(self):
        """Test basic reviews URL."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        url = scraper._build_reviews_url("B08N5WRWNW")

        assert "product-reviews/B08N5WRWNW" in url
        assert "pageNumber=1" in url

    def test_build_reviews_url_pagination(self):
        """Test reviews URL with pagination."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        url = scraper._build_reviews_url("B08N5WRWNW", page=3)

        assert "pageNumber=3" in url

    def test_build_reviews_url_sort(self):
        """Test reviews URL with sort."""
        from sentimatrix.providers.scrapers.platforms.amazon import (
            AmazonScraper,
            SortOrder,
        )

        scraper = AmazonScraper()

        url = scraper._build_reviews_url("B08N5WRWNW", sort_by=SortOrder.HELPFUL)

        assert "sortBy=helpful" in url

    def test_build_reviews_url_filter(self):
        """Test reviews URL with filter."""
        from sentimatrix.providers.scrapers.platforms.amazon import (
            AmazonScraper,
            ReviewFilter,
        )

        scraper = AmazonScraper()

        url = scraper._build_reviews_url("B08N5WRWNW", filter_by=ReviewFilter.VERIFIED)

        assert "reviewerType=avp_only_reviews" in url


class TestAmazonScraperScrape:
    """Test scraping functionality."""

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self):
        """Test scraping without initialization raises error."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper
        from sentimatrix.core.exceptions import ProviderInitializationError

        scraper = AmazonScraper()

        with pytest.raises(ProviderInitializationError):
            await scraper.scrape_reviews("B08N5WRWNW")

    @pytest.mark.asyncio
    async def test_scrape_invalid_asin(self):
        """Test scraping with invalid ASIN."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()
        scraper._initialized = True

        with pytest.raises(ValueError, match="Invalid ASIN"):
            await scraper.scrape_reviews("invalid")


class TestAmazonScraperParsing:
    """Test HTML parsing."""

    def test_parse_single_review(self):
        """Test parsing a single review element."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        # Create mock BeautifulSoup element
        mock_element = MagicMock()

        # Mock review body
        mock_body = MagicMock()
        mock_body.get_text.return_value = "This is a great product!"
        mock_element.select_one.side_effect = lambda sel: {
            "[data-hook='review-body']": mock_body,
            "[data-hook='review-star-rating']": MagicMock(get_text=MagicMock(return_value="5.0 out of 5 stars")),
            ".a-profile-name": MagicMock(get_text=MagicMock(return_value="TestUser")),
            "[data-hook='review-date']": MagicMock(get_text=MagicMock(return_value="Reviewed in the United States on January 15, 2024")),
            "[data-hook='review-title']": MagicMock(get_text=MagicMock(return_value="Amazing!")),
            "[data-hook='avp-badge']": MagicMock(),
            "[data-hook='helpful-vote-statement']": MagicMock(get_text=MagicMock(return_value="10 people found this helpful")),
        }.get(sel)

        review = scraper._parse_single_review(mock_element, "B08N5WRWNW")

        assert review is not None
        assert "great product" in review.text
        assert review.rating == 5.0
        assert review.author == "TestUser"
        assert review.metadata["verified_purchase"] is True
        assert review.metadata["helpful_votes"] == 10

    def test_parse_review_no_body(self):
        """Test parsing review with no body returns None."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()

        mock_element = MagicMock()
        mock_element.select_one.return_value = None

        review = scraper._parse_single_review(mock_element, "B08N5WRWNW")

        assert review is None


class TestAmazonScraperClose:
    """Test scraper close method."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing scraper."""
        from sentimatrix.providers.scrapers.platforms.amazon import AmazonScraper

        scraper = AmazonScraper()
        scraper._initialized = True
        scraper._playwright_scraper = AsyncMock()
        scraper._httpx_scraper = AsyncMock()

        await scraper.close()

        assert not scraper._initialized
        assert scraper._playwright_scraper is None
        assert scraper._httpx_scraper is None
