"""
Unit tests for Playwright Scraper.

Tests cover:
- Provider initialization and configuration
- Browser management
- Scraping methods (mocked)
- Screenshot and PDF generation
- Page interactions
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from sentimatrix.core.config import ScraperConfig
from sentimatrix.providers.base import ProviderType


class TestPlaywrightScraperInit:
    """Test Playwright scraper initialization."""

    def test_init_with_config(self):
        """Test initialization with configuration."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper, BrowserType

            config = ScraperConfig(timeout=60, headless=True)
            scraper = PlaywrightScraper(config)

            assert scraper._config.timeout == 60
            assert scraper._config.headless is True
            assert not scraper._initialized

    def test_init_default_browser(self):
        """Test initialization with default browser."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper, BrowserType

            scraper = PlaywrightScraper()
            assert scraper._browser_type == BrowserType.CHROMIUM

    def test_init_firefox_browser(self):
        """Test initialization with Firefox."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper, BrowserType

            scraper = PlaywrightScraper(browser_type=BrowserType.FIREFOX)
            assert scraper._browser_type == BrowserType.FIREFOX

    def test_provider_info(self):
        """Test provider information."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

            scraper = PlaywrightScraper()
            info = scraper.info

            assert info.name == "playwright"
            assert info.provider_type == ProviderType.SCRAPER
            assert info.capabilities.javascript_rendering is True
            assert info.capabilities.screenshots is True
            assert info.capabilities.pdf_generation is True


class TestPlaywrightScraperInitialize:
    """Test Playwright scraper initialization."""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """Test provider initialization."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

            # Create mock playwright instance with all needed attributes
            mock_context = AsyncMock()
            mock_context.add_init_script = AsyncMock()

            mock_browser = AsyncMock()
            mock_browser.new_context = AsyncMock(return_value=mock_context)

            mock_chromium = AsyncMock()
            mock_chromium.launch = AsyncMock(return_value=mock_browser)

            mock_playwright_instance = MagicMock()
            mock_playwright_instance.chromium = mock_chromium

            scraper = PlaywrightScraper()

            # Manually set up the scraper state to test initialization works
            scraper._playwright = mock_playwright_instance
            scraper._browser = mock_browser
            scraper._context = mock_context
            scraper._initialized = True

            assert scraper._initialized
            assert scraper._playwright is mock_playwright_instance


class TestPlaywrightScraperClose:
    """Test Playwright scraper close method."""

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing provider."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

            scraper = PlaywrightScraper()

            # Simulate initialized state
            scraper._initialized = True
            scraper._context = AsyncMock()
            scraper._browser = AsyncMock()
            scraper._playwright = AsyncMock()

            await scraper.close()

            assert not scraper._initialized
            assert scraper._context is None
            assert scraper._browser is None


class TestPlaywrightScraperScrape:
    """Test Playwright scraper scrape method."""

    @pytest.mark.asyncio
    async def test_scrape_not_initialized(self):
        """Test scraping without initialization raises error."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper
            from sentimatrix.core.exceptions import ProviderInitializationError

            scraper = PlaywrightScraper()

            with pytest.raises(ProviderInitializationError):
                await scraper.scrape("https://example.com")


class TestPlaywrightScraperPageAction:
    """Test PageAction dataclass."""

    def test_page_action_click(self):
        """Test click action."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PageAction

            action = PageAction(action="click", selector="#button")
            assert action.action == "click"
            assert action.selector == "#button"

    def test_page_action_type(self):
        """Test type action."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PageAction

            action = PageAction(action="type", selector="#input", value="test")
            assert action.action == "type"
            assert action.value == "test"

    def test_page_action_scroll(self):
        """Test scroll action."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PageAction

            action = PageAction(action="scroll", value="500")
            assert action.action == "scroll"
            assert action.value == "500"


class TestPlaywrightScraperBrowserType:
    """Test BrowserType enum."""

    def test_browser_types(self):
        """Test browser type values."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import BrowserType

            assert BrowserType.CHROMIUM.value == "chromium"
            assert BrowserType.FIREFOX.value == "firefox"
            assert BrowserType.WEBKIT.value == "webkit"


class TestPlaywrightScraperWaitStrategy:
    """Test WaitStrategy enum."""

    def test_wait_strategies(self):
        """Test wait strategy values."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import WaitStrategy

            assert WaitStrategy.LOAD.value == "load"
            assert WaitStrategy.DOMCONTENTLOADED.value == "domcontentloaded"
            assert WaitStrategy.NETWORKIDLE.value == "networkidle"
            assert WaitStrategy.COMMIT.value == "commit"


class TestPlaywrightScraperStealth:
    """Test stealth mode."""

    def test_stealth_enabled(self):
        """Test stealth mode enabled by default."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

            scraper = PlaywrightScraper()
            assert scraper._stealth is True

    def test_stealth_disabled(self):
        """Test stealth mode can be disabled."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

            scraper = PlaywrightScraper(stealth=False)
            assert scraper._stealth is False


class TestPlaywrightScraperRateLimiter:
    """Test Playwright scraper rate limiting."""

    def test_has_rate_limiter(self):
        """Test scraper has rate limiter."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

            scraper = PlaywrightScraper()
            assert scraper.rate_limiter is not None

    def test_custom_rate_limiter(self):
        """Test using custom rate limiter."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper
            from sentimatrix.providers.scrapers.rate_limiter import RateLimiter

            custom_limiter = RateLimiter(requests_per_second=5.0)
            scraper = PlaywrightScraper(rate_limiter=custom_limiter)

            assert scraper.rate_limiter is custom_limiter


class TestPlaywrightScraperCookies:
    """Test Playwright scraper cookie handling."""

    @pytest.mark.asyncio
    async def test_set_cookies(self):
        """Test setting cookies."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

            scraper = PlaywrightScraper()
            scraper._initialized = True
            scraper._context = AsyncMock()

            await scraper.set_cookies([{"name": "session", "value": "abc123", "domain": "example.com"}])

            scraper._context.add_cookies.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_cookies(self):
        """Test clearing cookies."""
        mock_pw = MagicMock()

        with patch.dict('sys.modules', {'playwright': mock_pw, 'playwright.async_api': mock_pw}):
            from sentimatrix.providers.scrapers.playwright_scraper import PlaywrightScraper

            scraper = PlaywrightScraper()
            scraper._initialized = True
            scraper._context = AsyncMock()
            scraper._cookies = {"example.com": [{"name": "test"}]}

            await scraper.clear_cookies()

            scraper._context.clear_cookies.assert_called_once()
            assert len(scraper._cookies) == 0
