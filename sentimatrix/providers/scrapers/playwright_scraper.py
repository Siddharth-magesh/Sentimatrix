"""
Sentimatrix Playwright Scraper Provider

Browser automation scraper using Playwright for JavaScript-rendered content.
Supports headless browsing, screenshots, and complex page interactions.

Features:
- Multi-browser support (Chromium, Firefox, WebKit)
- JavaScript rendering
- Screenshots and PDF generation
- Page interactions (click, type, scroll)
- Network interception
- Stealth mode for anti-detection
- Rate limiting integration

Example:
    >>> config = ScraperConfig(headless=True, timeout=30)
    >>> async with PlaywrightScraper(config) as scraper:
    ...     content = await scraper.scrape("https://example.com")
    ...     await scraper.screenshot("https://example.com", "screenshot.png")
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from sentimatrix.core.config import ScraperConfig
from sentimatrix.core.exceptions import (
    PlaywrightError,
    ScraperError,
    ScraperTimeoutError,
    ScraperConnectionError,
    RateLimitError,
)
from sentimatrix.providers.base import (
    BaseScraperProvider,
    ProviderCapabilities,
    ProviderInfo,
    ProviderType,
    Review,
    ScrapedContent,
    register_provider,
)
from sentimatrix.providers.scrapers.rate_limiter import RateLimiter, RateLimitStrategy
from sentimatrix.providers.scrapers.utils import (
    ProxyManager,
    UserAgentRotator,
    DeviceType,
    extract_domain,
)


# Lazy import for playwright
_playwright_module = None
_playwright_async = None


def _get_playwright():
    """Lazy import playwright."""
    global _playwright_module, _playwright_async
    if _playwright_module is None:
        try:
            from playwright.async_api import async_playwright, Playwright
            _playwright_module = True
            _playwright_async = async_playwright
        except ImportError as e:
            raise ImportError(
                "playwright is required for PlaywrightScraper. "
                "Install with: pip install playwright && playwright install"
            ) from e
    return _playwright_async


class BrowserType(str, Enum):
    """Supported browser types."""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class WaitStrategy(str, Enum):
    """Page wait strategies."""

    LOAD = "load"
    DOMCONTENTLOADED = "domcontentloaded"
    NETWORKIDLE = "networkidle"
    COMMIT = "commit"


@dataclass
class PageAction:
    """Represents a page interaction action."""

    action: str  # click, type, scroll, wait, evaluate
    selector: Optional[str] = None
    value: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class PlaywrightScraper(BaseScraperProvider):
    """
    Browser automation scraper using Playwright.

    Provides full browser automation for scraping JavaScript-rendered
    content with support for:
    - Multiple browser engines (Chromium, Firefox, WebKit)
    - Headless and headed modes
    - Page interactions (click, type, scroll)
    - Screenshots and PDF generation
    - Network request interception
    - Cookie and session management
    - Stealth mode for anti-detection

    Best for: JavaScript-heavy sites, SPAs, sites requiring login.
    """

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        browser_type: BrowserType = BrowserType.CHROMIUM,
        rate_limiter: Optional[RateLimiter] = None,
        proxy_manager: Optional[ProxyManager] = None,
        user_agent_rotator: Optional[UserAgentRotator] = None,
        stealth: bool = True,
    ) -> None:
        """
        Initialize Playwright scraper.

        Args:
            config: Scraper configuration
            browser_type: Browser engine to use
            rate_limiter: Optional rate limiter
            proxy_manager: Optional proxy manager
            user_agent_rotator: Optional user agent rotator
            stealth: Enable stealth mode for anti-detection
        """
        super().__init__(config)
        self._config: ScraperConfig = config or ScraperConfig()
        self._browser_type = browser_type
        self._stealth = stealth

        # Playwright objects
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None

        # Rate limiting
        self._rate_limiter = rate_limiter or RateLimiter(
            config=self._config.rate_limit,
            strategy=RateLimitStrategy.TOKEN_BUCKET,
            requests_per_second=self._config.rate_limit.requests_per_second,
            per_domain=True,
        )

        # Proxy management
        self._proxy_manager = proxy_manager
        if self._config.proxy.enabled and not proxy_manager:
            self._proxy_manager = ProxyManager(config=self._config.proxy)

        # User agent rotation
        self._ua_rotator = user_agent_rotator or UserAgentRotator(
            device_type=DeviceType.DESKTOP
        )

        # Stored cookies per domain
        self._cookies: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="playwright",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Browser automation scraper using Playwright for JS-rendered content",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                screenshots=True,
                pdf_generation=True,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    async def initialize(self) -> None:
        """Initialize Playwright and launch browser."""
        if self._initialized:
            return

        async_playwright = _get_playwright()

        try:
            self._playwright = await async_playwright().start()

            # Get browser launcher
            if self._browser_type == BrowserType.CHROMIUM:
                launcher = self._playwright.chromium
            elif self._browser_type == BrowserType.FIREFOX:
                launcher = self._playwright.firefox
            else:
                launcher = self._playwright.webkit

            # Build launch arguments
            launch_args: Dict[str, Any] = {
                "headless": self._config.headless,
            }

            # Add proxy if configured
            if self._proxy_manager:
                proxy_config = self._proxy_manager.get_proxy_for_playwright()
                if proxy_config:
                    launch_args["proxy"] = proxy_config

            # Stealth args for Chromium
            if self._stealth and self._browser_type == BrowserType.CHROMIUM:
                launch_args["args"] = [
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-site-isolation-trials",
                ]

            self._browser = await launcher.launch(**launch_args)

            # Create default context
            context_options = {
                "viewport": {
                    "width": self._config.viewport_width,
                    "height": self._config.viewport_height,
                },
                "user_agent": self._config.user_agent or self._ua_rotator.get_user_agent(),
            }

            # Additional stealth options
            if self._stealth:
                context_options["java_script_enabled"] = True
                context_options["bypass_csp"] = True

            self._context = await self._browser.new_context(**context_options)

            # Apply stealth scripts if enabled
            if self._stealth:
                await self._apply_stealth_scripts()

            self._initialized = True

        except Exception as e:
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
            raise PlaywrightError(
                action="initialize",
                message=f"Failed to initialize Playwright: {e}",
            ) from e

    async def _apply_stealth_scripts(self) -> None:
        """Apply stealth scripts to avoid detection."""
        # Add init script to mask webdriver
        stealth_js = """
        // Overwrite navigator.webdriver
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });

        // Overwrite plugins
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5]
        });

        // Overwrite languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en']
        });

        // Chrome specific
        window.chrome = {
            runtime: {}
        };

        // Permissions
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) => (
            parameters.name === 'notifications' ?
            Promise.resolve({ state: Notification.permission }) :
            originalQuery(parameters)
        );
        """

        await self._context.add_init_script(stealth_js)

    async def close(self) -> None:
        """Close browser and cleanup."""
        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self._initialized = False

    async def scrape(
        self,
        url: str,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        wait_strategy: WaitStrategy = WaitStrategy.NETWORKIDLE,
        actions: Optional[List[PageAction]] = None,
        **kwargs: Any,
    ) -> ScrapedContent:
        """
        Scrape content from a URL with full browser rendering.

        Args:
            url: URL to scrape
            wait_for: CSS selector to wait for before scraping
            timeout: Page load timeout (overrides config)
            wait_strategy: Page wait strategy
            actions: List of page actions to perform
            **kwargs: Additional arguments

        Returns:
            ScrapedContent with rendered page content

        Raises:
            ScraperError: If scraping fails
            ScraperTimeoutError: If page load times out
        """
        self._ensure_initialized()

        domain = extract_domain(url)
        start_time = time.perf_counter()

        # Rate limiting
        await self._rate_limiter.acquire(domain)

        timeout_ms = (timeout or self._config.timeout) * 1000
        page = None
        proxy_url = None

        try:
            # Create new page
            page = await self._context.new_page()

            # Set extra headers if provided
            if "headers" in kwargs:
                await page.set_extra_http_headers(kwargs["headers"])

            # Load stored cookies for this domain
            if domain in self._cookies:
                await self._context.add_cookies(self._cookies[domain])

            # Navigate to URL
            response = await page.goto(
                url,
                timeout=timeout_ms,
                wait_until=wait_strategy.value,
            )

            # Wait for specific selector if provided
            if wait_for or self._config.wait_for_selector:
                selector = wait_for or self._config.wait_for_selector
                await page.wait_for_selector(selector, timeout=timeout_ms)

            # Perform actions if provided
            if actions:
                await self._perform_actions(page, actions, timeout_ms)

            # Add random delay to appear more human
            await asyncio.sleep(random.uniform(0.5, 1.5))

            # Get page content
            html = await page.content()
            title = await page.title()

            # Get text content
            text_content = await page.evaluate("""
                () => {
                    // Remove script and style elements
                    const scripts = document.querySelectorAll('script, style, noscript, iframe');
                    scripts.forEach(s => s.remove());
                    return document.body.innerText;
                }
            """)

            # Store cookies
            cookies = await self._context.cookies()
            domain_cookies = [c for c in cookies if domain in c.get("domain", "")]
            if domain_cookies:
                self._cookies[domain] = domain_cookies

            response_time = (time.perf_counter() - start_time) * 1000

            # Get response info
            status_code = response.status if response else 200
            headers = dict(response.headers) if response else {}

            return ScrapedContent(
                url=page.url,
                title=title,
                content=text_content or "",
                html=html,
                status_code=status_code,
                response_time_ms=response_time,
                headers=headers,
                provider="playwright",
                proxy_used=proxy_url,
                user_agent=self._config.user_agent or self._ua_rotator.get_user_agent(),
            )

        except Exception as e:
            error_msg = str(e).lower()

            if "timeout" in error_msg:
                raise ScraperTimeoutError(
                    url=url,
                    timeout=timeout or self._config.timeout,
                ) from e

            if "net::" in error_msg or "connection" in error_msg:
                raise ScraperConnectionError(
                    url=url,
                    reason=str(e),
                ) from e

            raise PlaywrightError(
                action="scrape",
                message=f"Failed to scrape {url}: {e}",
            ) from e

        finally:
            if page:
                await page.close()

    async def _perform_actions(
        self,
        page: Any,
        actions: List[PageAction],
        timeout_ms: int,
    ) -> None:
        """Perform page actions."""
        for action in actions:
            if action.action == "click":
                if action.selector:
                    await page.click(action.selector, timeout=timeout_ms)

            elif action.action == "type":
                if action.selector and action.value:
                    await page.fill(action.selector, action.value)

            elif action.action == "scroll":
                if action.value:
                    await page.evaluate(f"window.scrollBy(0, {action.value})")
                else:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")

            elif action.action == "wait":
                if action.selector:
                    await page.wait_for_selector(action.selector, timeout=timeout_ms)
                elif action.value:
                    await asyncio.sleep(float(action.value))

            elif action.action == "evaluate":
                if action.value:
                    await page.evaluate(action.value)

            elif action.action == "press":
                if action.value:
                    await page.keyboard.press(action.value)

            elif action.action == "hover":
                if action.selector:
                    await page.hover(action.selector)

            # Small delay between actions
            await asyncio.sleep(random.uniform(0.1, 0.3))

    async def scrape_reviews(
        self,
        url: str,
        limit: int = 100,
        sort_by: Optional[str] = None,
        selectors: Optional[Dict[str, str]] = None,
        scroll_for_more: bool = True,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Extract reviews from a URL with JavaScript rendering.

        Args:
            url: URL to scrape reviews from
            limit: Maximum number of reviews
            sort_by: Sort option to click (CSS selector)
            selectors: CSS selectors for review elements
            scroll_for_more: Scroll to load more reviews
            **kwargs: Additional arguments

        Returns:
            List of Review objects
        """
        self._ensure_initialized()

        if not selectors:
            raise ValueError(
                "selectors dict required with at least 'container' and 'text' keys"
            )

        domain = extract_domain(url)
        await self._rate_limiter.acquire(domain)

        timeout_ms = (kwargs.get("timeout") or self._config.timeout) * 1000
        page = None

        try:
            page = await self._context.new_page()

            await page.goto(
                url,
                timeout=timeout_ms,
                wait_until="networkidle",
            )

            # Click sort option if provided
            if sort_by:
                try:
                    await page.click(sort_by, timeout=5000)
                    await asyncio.sleep(1)
                except Exception:
                    pass  # Ignore sort failures

            reviews: List[Review] = []
            container_selector = selectors.get("container", "div.review")
            scroll_attempts = 0
            max_scroll_attempts = 10

            while len(reviews) < limit and scroll_attempts < max_scroll_attempts:
                # Extract reviews
                containers = await page.query_selector_all(container_selector)

                for i, container in enumerate(containers):
                    if len(reviews) >= limit:
                        break

                    # Check if we already have this review
                    review_id = f"{domain}-{i}"
                    if any(r.id == review_id for r in reviews):
                        continue

                    # Extract text
                    text = ""
                    if "text" in selectors:
                        text_elem = await container.query_selector(selectors["text"])
                        if text_elem:
                            text = await text_elem.inner_text()

                    if not text:
                        continue

                    # Extract rating
                    rating = None
                    if "rating" in selectors:
                        rating_elem = await container.query_selector(selectors["rating"])
                        if rating_elem:
                            try:
                                rating_text = await rating_elem.inner_text()
                                import re
                                match = re.search(r"(\d+(?:\.\d+)?)", rating_text)
                                if match:
                                    rating = float(match.group(1))
                            except Exception:
                                pass

                    # Extract author
                    author = None
                    if "author" in selectors:
                        author_elem = await container.query_selector(selectors["author"])
                        if author_elem:
                            author = await author_elem.inner_text()

                    reviews.append(Review(
                        id=review_id,
                        text=text.strip(),
                        source=url,
                        platform=domain,
                        author=author,
                        rating=rating,
                    ))

                # Scroll for more if needed
                if scroll_for_more and len(reviews) < limit:
                    previous_count = len(reviews)
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(2)  # Wait for content to load
                    scroll_attempts += 1

                    # Check if new reviews loaded
                    new_containers = await page.query_selector_all(container_selector)
                    if len(new_containers) == len(containers):
                        # No new content, try clicking "load more" button
                        if "load_more" in selectors:
                            try:
                                await page.click(selectors["load_more"], timeout=3000)
                                await asyncio.sleep(2)
                            except Exception:
                                break
                        else:
                            break
                else:
                    break

            return reviews[:limit]

        except Exception as e:
            raise PlaywrightError(
                action="scrape_reviews",
                message=f"Failed to scrape reviews from {url}: {e}",
            ) from e

        finally:
            if page:
                await page.close()

    async def screenshot(
        self,
        url: str,
        path: str,
        full_page: bool = True,
        wait_for: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Take a screenshot of a page.

        Args:
            url: URL to screenshot
            path: Output file path
            full_page: Capture full page or viewport only
            wait_for: CSS selector to wait for
            **kwargs: Additional arguments

        Returns:
            Path to saved screenshot
        """
        self._ensure_initialized()

        domain = extract_domain(url)
        await self._rate_limiter.acquire(domain)

        timeout_ms = (kwargs.get("timeout") or self._config.timeout) * 1000
        page = None

        try:
            page = await self._context.new_page()

            await page.goto(
                url,
                timeout=timeout_ms,
                wait_until="networkidle",
            )

            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout_ms)

            # Ensure directory exists
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

            await page.screenshot(path=path, full_page=full_page)

            return path

        except Exception as e:
            raise PlaywrightError(
                action="screenshot",
                message=f"Failed to take screenshot of {url}: {e}",
            ) from e

        finally:
            if page:
                await page.close()

    async def pdf(
        self,
        url: str,
        path: str,
        wait_for: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate PDF from a page.

        Args:
            url: URL to convert
            path: Output PDF path
            wait_for: CSS selector to wait for
            **kwargs: Additional arguments

        Returns:
            Path to saved PDF
        """
        self._ensure_initialized()

        # PDF only works with Chromium
        if self._browser_type != BrowserType.CHROMIUM:
            raise PlaywrightError(
                action="pdf",
                message="PDF generation only supported with Chromium browser",
            )

        domain = extract_domain(url)
        await self._rate_limiter.acquire(domain)

        timeout_ms = (kwargs.get("timeout") or self._config.timeout) * 1000
        page = None

        try:
            page = await self._context.new_page()

            await page.goto(
                url,
                timeout=timeout_ms,
                wait_until="networkidle",
            )

            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout_ms)

            # Ensure directory exists
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

            await page.pdf(path=path, format="A4")

            return path

        except Exception as e:
            raise PlaywrightError(
                action="pdf",
                message=f"Failed to generate PDF from {url}: {e}",
            ) from e

        finally:
            if page:
                await page.close()

    async def evaluate(
        self,
        url: str,
        script: str,
        wait_for: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Navigate to URL and evaluate JavaScript.

        Args:
            url: URL to navigate to
            script: JavaScript to evaluate
            wait_for: CSS selector to wait for
            **kwargs: Additional arguments

        Returns:
            Result of JavaScript evaluation
        """
        self._ensure_initialized()

        domain = extract_domain(url)
        await self._rate_limiter.acquire(domain)

        timeout_ms = (kwargs.get("timeout") or self._config.timeout) * 1000
        page = None

        try:
            page = await self._context.new_page()

            await page.goto(
                url,
                timeout=timeout_ms,
                wait_until="networkidle",
            )

            if wait_for:
                await page.wait_for_selector(wait_for, timeout=timeout_ms)

            result = await page.evaluate(script)
            return result

        except Exception as e:
            raise PlaywrightError(
                action="evaluate",
                message=f"Failed to evaluate script on {url}: {e}",
            ) from e

        finally:
            if page:
                await page.close()

    async def new_page(self) -> Any:
        """
        Create a new page for manual control.

        Returns:
            Playwright Page object

        Note:
            Caller is responsible for closing the page.
        """
        self._ensure_initialized()
        return await self._context.new_page()

    def set_user_agent(self, user_agent: str) -> None:
        """Set user agent for future pages."""
        # This will be used when creating new contexts
        self._config = self._config.model_copy(update={"user_agent": user_agent})

    async def set_cookies(self, cookies: List[Dict[str, Any]]) -> None:
        """Set cookies in the browser context."""
        self._ensure_initialized()
        await self._context.add_cookies(cookies)

    async def clear_cookies(self) -> None:
        """Clear all cookies."""
        self._ensure_initialized()
        await self._context.clear_cookies()
        self._cookies.clear()

    @property
    def rate_limiter(self) -> RateLimiter:
        """Get the rate limiter."""
        return self._rate_limiter

    @property
    def proxy_manager(self) -> Optional[ProxyManager]:
        """Get the proxy manager."""
        return self._proxy_manager

    @property
    def browser_type(self) -> BrowserType:
        """Get the browser type."""
        return self._browser_type


# Register provider
register_provider("playwright", ProviderType.SCRAPER, PlaywrightScraper)
