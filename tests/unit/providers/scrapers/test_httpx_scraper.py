"""
Unit tests for HTTPX Scraper.

Tests cover:
- Provider initialization and configuration
- Scraping methods (mocked HTTP)
- Rate limiting integration
- Proxy support
- User agent rotation
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from sentimatrix.core.config import ScraperConfig
from sentimatrix.providers.base import ProviderType, ScrapedContent


# Mock response class
@dataclass
class MockResponse:
    status_code: int = 200
    text: str = "<html><head><title>Test Page</title></head><body>Test content</body></html>"
    url: str = "https://example.com"
    headers: dict = None
    cookies: dict = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {"content-type": "text/html"}
        if self.cookies is None:
            self.cookies = {}

    def items(self):
        return self.cookies.items()

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestHTTPXScraperInit:
    """Test HTTPX scraper initialization."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            config = ScraperConfig(timeout=60, headless=True)
            scraper = HTTPXScraper(config)

            assert scraper._config.timeout == 60
            assert not scraper._initialized

    def test_init_default_config(self):
        """Test initialization with default config."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            assert scraper._config is not None

    def test_provider_info(self):
        """Test provider information."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            info = scraper.info

            assert info.name == "httpx"
            assert info.provider_type == ProviderType.SCRAPER
            assert info.capabilities.javascript_rendering is False
            assert info.capabilities.proxy_support is True


class TestHTTPXScraperInitialize:
    """Test HTTPX scraper initialization."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test provider initialization."""
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=AsyncMock())
        mock_httpx.Timeout = MagicMock()
        mock_httpx.Limits = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            # Reset the cached module
            import sentimatrix.providers.scrapers.httpx_scraper as module
            module._httpx = None

            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            await scraper.initialize()

            assert scraper._initialized
            mock_httpx.AsyncClient.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Test initialization is idempotent."""
        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=AsyncMock())
        mock_httpx.Timeout = MagicMock()
        mock_httpx.Limits = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            import sentimatrix.providers.scrapers.httpx_scraper as module
            module._httpx = None

            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            await scraper.initialize()
            await scraper.initialize()

            # Should only create client once
            assert mock_httpx.AsyncClient.call_count == 1


class TestHTTPXScraperScrape:
    """Test HTTPX scraper scrape method."""

    @pytest.mark.asyncio
    async def test_scrape_basic(self):
        """Test basic scraping."""
        mock_response = MockResponse()
        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.aclose = AsyncMock()

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
        mock_httpx.Timeout = MagicMock()
        mock_httpx.Limits = MagicMock()
        mock_httpx.TimeoutException = Exception
        mock_httpx.HTTPStatusError = Exception
        mock_httpx.ConnectError = Exception

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.find.return_value = MagicMock(get_text=MagicMock(return_value="Test Page"))
        mock_soup.__call__ = MagicMock(return_value=[])
        mock_soup.get_text = MagicMock(return_value="Test content")

        mock_bs4_class = MagicMock(return_value=mock_soup)
        mock_bs4_module = MagicMock()
        mock_bs4_module.BeautifulSoup = mock_bs4_class

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4_module}):
            import sentimatrix.providers.scrapers.httpx_scraper as module
            module._httpx = None
            module._bs4 = None

            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            await scraper.initialize()

            content = await scraper.scrape("https://example.com")

            assert isinstance(content, ScrapedContent)
            assert content.provider == "httpx"
            assert content.status_code == 200

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self):
        """Test scraping without initialization raises error."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper
            from sentimatrix.core.exceptions import ProviderInitializationError

            scraper = HTTPXScraper()

            with pytest.raises(ProviderInitializationError):
                await scraper.scrape("https://example.com")


class TestHTTPXScraperClose:
    """Test HTTPX scraper close method."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing provider."""
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
        mock_httpx.Timeout = MagicMock()
        mock_httpx.Limits = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            import sentimatrix.providers.scrapers.httpx_scraper as module
            module._httpx = None

            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            await scraper.initialize()
            await scraper.close()

            assert not scraper._initialized
            mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient = MagicMock(return_value=mock_client)
        mock_httpx.Timeout = MagicMock()
        mock_httpx.Limits = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            import sentimatrix.providers.scrapers.httpx_scraper as module
            module._httpx = None

            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            async with HTTPXScraper() as scraper:
                assert scraper._initialized

            assert not scraper._initialized


class TestHTTPXScraperHeaders:
    """Test HTTPX scraper header handling."""

    def test_get_headers_default(self):
        """Test default headers include user agent."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            headers = scraper._get_headers()

            assert "User-Agent" in headers
            assert "Accept" in headers
            assert "Accept-Language" in headers

    def test_get_headers_custom(self):
        """Test custom headers merge."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            headers = scraper._get_headers({"X-Custom": "value"})

            assert headers["X-Custom"] == "value"
            assert "User-Agent" in headers


class TestHTTPXScraperCookies:
    """Test HTTPX scraper cookie handling."""

    def test_set_cookies(self):
        """Test setting session cookies."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            scraper.set_cookies({"session": "abc123"})

            assert scraper._cookies["session"] == "abc123"

    def test_clear_cookies(self):
        """Test clearing session cookies."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            scraper.set_cookies({"session": "abc123"})
            scraper.clear_cookies()

            assert len(scraper._cookies) == 0


class TestHTTPXScraperRateLimiter:
    """Test HTTPX scraper rate limiting."""

    def test_has_rate_limiter(self):
        """Test scraper has rate limiter."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper

            scraper = HTTPXScraper()
            assert scraper.rate_limiter is not None

    def test_custom_rate_limiter(self):
        """Test using custom rate limiter."""
        mock_httpx = MagicMock()
        mock_bs4 = MagicMock()

        with patch.dict('sys.modules', {'httpx': mock_httpx, 'bs4': mock_bs4}):
            from sentimatrix.providers.scrapers.httpx_scraper import HTTPXScraper
            from sentimatrix.providers.scrapers.rate_limiter import RateLimiter

            custom_limiter = RateLimiter(requests_per_second=5.0)
            scraper = HTTPXScraper(rate_limiter=custom_limiter)

            assert scraper.rate_limiter is custom_limiter
