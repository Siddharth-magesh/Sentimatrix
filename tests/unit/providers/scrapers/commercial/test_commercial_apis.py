"""
Unit tests for Commercial Scraping API clients.

Tests cover:
- ScraperAPIClient
- ApifyClient
- BrightDataClient
- OxylabsClient
- ZyteClient
- ScrapingBeeClient
- ScrapingAntClient
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from sentimatrix.providers.base import ProviderType


# ============================================================================
# Base Config Tests
# ============================================================================

class TestCommercialAPIConfig:
    """Test base configuration classes."""

    def test_base_config_defaults(self):
        """Test CommercialAPIConfig defaults."""
        from sentimatrix.providers.scrapers.commercial.base import CommercialAPIConfig

        config = CommercialAPIConfig()
        assert config.api_key is None
        assert config.timeout == 60
        assert config.retries == 3
        assert config.render_js is False
        assert config.requests_per_second == 5.0

    def test_device_emulation_enum(self):
        """Test DeviceEmulation enum."""
        from sentimatrix.providers.scrapers.commercial.base import DeviceEmulation

        assert DeviceEmulation.DESKTOP == "desktop"
        assert DeviceEmulation.MOBILE == "mobile"
        assert DeviceEmulation.TABLET == "tablet"

    def test_output_format_enum(self):
        """Test OutputFormat enum."""
        from sentimatrix.providers.scrapers.commercial.base import OutputFormat

        assert OutputFormat.HTML == "html"
        assert OutputFormat.JSON == "json"
        assert OutputFormat.MARKDOWN == "markdown"

    def test_scrape_result_dataclass(self):
        """Test ScrapeResult dataclass."""
        from sentimatrix.providers.scrapers.commercial.base import ScrapeResult

        result = ScrapeResult(
            url="https://example.com",
            content="Test content",
            status_code=200,
            provider="test",
        )

        assert result.url == "https://example.com"
        assert result.content == "Test content"
        assert result.status_code == 200
        assert result.credits_used == 1


# ============================================================================
# ScraperAPI Tests
# ============================================================================

class TestScraperAPIClient:
    """Test ScraperAPI client."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.commercial import ScraperAPIClient, ScraperAPIConfig

        config = ScraperAPIConfig(api_key="test_key", render_js=True)
        client = ScraperAPIClient(config)

        assert client._scraper_config.api_key == "test_key"
        assert client._scraper_config.render_js is True
        assert client.SERVICE_NAME == "scraperapi"

    def test_init_with_api_key(self):
        """Test initialization with api_key parameter."""
        from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

        client = ScraperAPIClient(api_key="direct_key")
        assert client._scraper_config.api_key == "direct_key"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

        client = ScraperAPIClient(api_key="test")
        info = client.info

        assert info.name == "scraperapi"
        assert info.provider_type == ProviderType.SCRAPER
        assert info.capabilities.javascript_rendering is True
        assert info.capabilities.proxy_support is True

    def test_base_url(self):
        """Test base URL."""
        from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

        assert ScraperAPIClient.BASE_URL == "https://api.scraperapi.com"

    def test_config_options(self):
        """Test ScraperAPI config options."""
        from sentimatrix.providers.scrapers.commercial import ScraperAPIConfig

        config = ScraperAPIConfig(
            api_key="key",
            premium_proxy=True,
            ultra_premium=True,
            country_code="us",
            autoparse=True,
        )

        assert config.premium_proxy is True
        assert config.ultra_premium is True
        assert config.country_code == "us"
        assert config.autoparse is True


# ============================================================================
# Apify Tests
# ============================================================================

class TestApifyClient:
    """Test Apify client."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.commercial import ApifyClient, ApifyConfig

        config = ApifyConfig(api_token="test_token")
        client = ApifyClient(config)

        assert client._apify_config.api_token == "test_token"
        assert client.SERVICE_NAME == "apify"

    def test_init_with_api_token(self):
        """Test initialization with api_token parameter."""
        from sentimatrix.providers.scrapers.commercial import ApifyClient

        client = ApifyClient(api_token="direct_token")
        assert client._apify_config.api_token == "direct_token"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.commercial import ApifyClient

        client = ApifyClient(api_token="test")
        info = client.info

        assert info.name == "apify"
        assert info.provider_type == ProviderType.SCRAPER
        assert info.capabilities.javascript_rendering is True
        assert "2000+" in info.description.lower() or "actor" in info.description.lower()

    def test_popular_actors(self):
        """Test popular actors dictionary."""
        from sentimatrix.providers.scrapers.commercial import POPULAR_ACTORS

        assert "web-scraper" in POPULAR_ACTORS
        assert "amazon" in POPULAR_ACTORS
        assert "youtube" in POPULAR_ACTORS
        assert POPULAR_ACTORS["web-scraper"] == "apify/web-scraper"

    def test_config_defaults(self):
        """Test Apify config defaults."""
        from sentimatrix.providers.scrapers.commercial import ApifyConfig

        config = ApifyConfig()
        assert config.default_actor_id == "apify/web-scraper"
        assert config.memory_mbytes == 1024
        assert config.timeout_secs == 300


# ============================================================================
# Bright Data Tests
# ============================================================================

class TestBrightDataClient:
    """Test Bright Data client."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.commercial import BrightDataClient, BrightDataConfig

        config = BrightDataConfig(api_token="test_token", customer_id="cust_123")
        client = BrightDataClient(config)

        assert client._bd_config.api_token == "test_token"
        assert client._bd_config.customer_id == "cust_123"
        assert client.SERVICE_NAME == "bright_data"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.commercial import BrightDataClient

        client = BrightDataClient(api_token="test")
        info = client.info

        assert info.name == "bright_data"
        assert info.provider_type == ProviderType.SCRAPER
        assert "72M+" in info.description or "enterprise" in info.description.lower()

    def test_zone_enum(self):
        """Test BrightDataZone enum."""
        from sentimatrix.providers.scrapers.commercial import BrightDataZone

        assert BrightDataZone.DATACENTER == "datacenter"
        assert BrightDataZone.RESIDENTIAL == "residential"
        assert BrightDataZone.MOBILE == "mobile"
        assert BrightDataZone.UNLOCKER == "unlocker"

    def test_proxy_url_generation(self):
        """Test proxy URL generation."""
        from sentimatrix.providers.scrapers.commercial import BrightDataClient, BrightDataConfig

        config = BrightDataConfig(
            api_token="test_pass",
            customer_id="cust_123",
        )
        client = BrightDataClient(config)

        proxy_url = client.get_proxy_url(country="us")
        assert "brd-customer-cust_123" in proxy_url
        assert "country-us" in proxy_url


# ============================================================================
# Oxylabs Tests
# ============================================================================

class TestOxylabsClient:
    """Test Oxylabs client."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.commercial import OxylabsClient, OxylabsConfig

        config = OxylabsConfig(username="user", password="pass")
        client = OxylabsClient(config)

        assert client._oxy_config.username == "user"
        assert client._oxy_config.password == "pass"
        assert client.SERVICE_NAME == "oxylabs"

    def test_init_with_credentials(self):
        """Test initialization with credential parameters."""
        from sentimatrix.providers.scrapers.commercial import OxylabsClient

        client = OxylabsClient(username="user", password="pass")
        assert client._oxy_config.username == "user"
        assert client._oxy_config.password == "pass"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.commercial import OxylabsClient

        client = OxylabsClient(username="user", password="pass")
        info = client.info

        assert info.name == "oxylabs"
        assert info.provider_type == ProviderType.SCRAPER
        assert info.capabilities.batch_processing is True

    def test_source_enum(self):
        """Test OxylabsSource enum."""
        from sentimatrix.providers.scrapers.commercial import OxylabsSource

        assert OxylabsSource.AMAZON_SEARCH == "amazon_search"
        assert OxylabsSource.GOOGLE_SEARCH == "google_search"
        assert OxylabsSource.WALMART == "walmart"
        assert OxylabsSource.UNIVERSAL == "universal"

    def test_api_urls(self):
        """Test API URLs."""
        from sentimatrix.providers.scrapers.commercial import OxylabsClient

        assert "realtime.oxylabs.io" in OxylabsClient.REALTIME_URL
        assert "data.oxylabs.io" in OxylabsClient.ASYNC_URL


# ============================================================================
# Zyte Tests
# ============================================================================

class TestZyteClient:
    """Test Zyte client."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.commercial import ZyteClient, ZyteConfig

        config = ZyteConfig(api_key="test_key")
        client = ZyteClient(config)

        assert client._zyte_config.api_key == "test_key"
        assert client.SERVICE_NAME == "zyte"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.commercial import ZyteClient

        client = ZyteClient(api_key="test")
        info = client.info

        assert info.name == "zyte"
        assert info.provider_type == ProviderType.SCRAPER
        assert "AI" in info.description or "extraction" in info.description.lower()

    def test_extraction_type_enum(self):
        """Test ZyteExtractionType enum."""
        from sentimatrix.providers.scrapers.commercial import ZyteExtractionType

        assert ZyteExtractionType.PRODUCT == "product"
        assert ZyteExtractionType.ARTICLE == "article"
        assert ZyteExtractionType.JOB_POSTING == "jobPosting"

    def test_action_enum(self):
        """Test ZyteAction enum."""
        from sentimatrix.providers.scrapers.commercial import ZyteAction

        assert ZyteAction.CLICK == "click"
        assert ZyteAction.SCROLL == "scroll"
        assert ZyteAction.WAIT_FOR_SELECTOR == "waitForSelector"

    def test_base_url(self):
        """Test base URL."""
        from sentimatrix.providers.scrapers.commercial import ZyteClient

        assert "api.zyte.com" in ZyteClient.BASE_URL


# ============================================================================
# ScrapingBee Tests
# ============================================================================

class TestScrapingBeeClient:
    """Test ScrapingBee client."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.commercial import ScrapingBeeClient, ScrapingBeeConfig

        config = ScrapingBeeConfig(api_key="test_key", render_js=True)
        client = ScrapingBeeClient(config)

        assert client._sb_config.api_key == "test_key"
        assert client._sb_config.render_js is True
        assert client.SERVICE_NAME == "scrapingbee"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.commercial import ScrapingBeeClient

        client = ScrapingBeeClient(api_key="test")
        info = client.info

        assert info.name == "scrapingbee"
        assert info.provider_type == ProviderType.SCRAPER
        assert info.capabilities.screenshots is True
        assert info.capabilities.pdf_generation is True

    def test_config_options(self):
        """Test ScrapingBee config options."""
        from sentimatrix.providers.scrapers.commercial import ScrapingBeeConfig

        config = ScrapingBeeConfig(
            api_key="key",
            premium_proxy=True,
            stealth_proxy=True,
            wait=5000,
            js_snippet="console.log('test')",
        )

        assert config.premium_proxy is True
        assert config.stealth_proxy is True
        assert config.wait == 5000
        assert config.js_snippet == "console.log('test')"


# ============================================================================
# ScrapingAnt Tests
# ============================================================================

class TestScrapingAntClient:
    """Test ScrapingAnt client."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        from sentimatrix.providers.scrapers.commercial import ScrapingAntClient, ScrapingAntConfig

        config = ScrapingAntConfig(api_key="test_key")
        client = ScrapingAntClient(config)

        assert client._sa_config.api_key == "test_key"
        assert client.SERVICE_NAME == "scrapingant"

    def test_provider_info(self):
        """Test provider information."""
        from sentimatrix.providers.scrapers.commercial import ScrapingAntClient

        client = ScrapingAntClient(api_key="test")
        info = client.info

        assert info.name == "scrapingant"
        assert info.provider_type == ProviderType.SCRAPER
        assert "budget" in info.description.lower()

    def test_cookie_class(self):
        """Test Cookie dataclass."""
        from sentimatrix.providers.scrapers.commercial import Cookie

        cookie = Cookie(name="session", value="abc123", domain=".example.com")

        assert cookie.name == "session"
        assert cookie.value == "abc123"
        assert cookie.domain == ".example.com"

        cookie_dict = cookie.to_dict()
        assert cookie_dict["name"] == "session"
        assert cookie_dict["value"] == "abc123"
        assert cookie_dict["domain"] == ".example.com"

    def test_config_defaults(self):
        """Test ScrapingAnt config defaults."""
        from sentimatrix.providers.scrapers.commercial import ScrapingAntConfig

        config = ScrapingAntConfig()
        assert config.browser is True
        assert config.proxy_type == "datacenter"


# ============================================================================
# Import Tests
# ============================================================================

class TestCommercialAPIImports:
    """Test that all commercial APIs are properly importable."""

    def test_all_clients_importable(self):
        """Test all clients can be imported."""
        from sentimatrix.providers.scrapers.commercial import (
            ScraperAPIClient,
            ApifyClient,
            BrightDataClient,
            OxylabsClient,
            ZyteClient,
            ScrapingBeeClient,
            ScrapingAntClient,
        )

        assert ScraperAPIClient is not None
        assert ApifyClient is not None
        assert BrightDataClient is not None
        assert OxylabsClient is not None
        assert ZyteClient is not None
        assert ScrapingBeeClient is not None
        assert ScrapingAntClient is not None

    def test_all_configs_importable(self):
        """Test all configs can be imported."""
        from sentimatrix.providers.scrapers.commercial import (
            ScraperAPIConfig,
            ApifyConfig,
            BrightDataConfig,
            OxylabsConfig,
            ZyteConfig,
            ScrapingBeeConfig,
            ScrapingAntConfig,
        )

        assert ScraperAPIConfig is not None
        assert ApifyConfig is not None
        assert BrightDataConfig is not None
        assert OxylabsConfig is not None
        assert ZyteConfig is not None
        assert ScrapingBeeConfig is not None
        assert ScrapingAntConfig is not None

    def test_base_classes_importable(self):
        """Test base classes can be imported."""
        from sentimatrix.providers.scrapers.commercial import (
            BaseCommercialClient,
            CommercialAPIConfig,
            ScrapeResult,
            OutputFormat,
            DeviceEmulation,
        )

        assert BaseCommercialClient is not None
        assert CommercialAPIConfig is not None
        assert ScrapeResult is not None
        assert OutputFormat is not None
        assert DeviceEmulation is not None


# ============================================================================
# Stats Tests
# ============================================================================

class TestClientStats:
    """Test client statistics tracking."""

    def test_stats_property(self):
        """Test stats property."""
        from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

        client = ScraperAPIClient(api_key="test")
        stats = client.stats

        assert "total_requests" in stats
        assert "total_credits" in stats
        assert "total_cost_usd" in stats
        assert stats["total_requests"] == 0

    def test_reset_stats(self):
        """Test reset_stats method."""
        from sentimatrix.providers.scrapers.commercial import ScraperAPIClient

        client = ScraperAPIClient(api_key="test")
        client._total_requests = 10
        client._total_credits = 50

        client.reset_stats()

        assert client._total_requests == 0
        assert client._total_credits == 0
