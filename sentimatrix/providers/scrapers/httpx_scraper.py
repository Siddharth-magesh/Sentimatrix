"""
Sentimatrix HTTPX Scraper Provider

Async HTTP scraper using HTTPX for high-concurrency web scraping.
Suitable for static HTML pages without JavaScript rendering.

Features:
- Async HTTP/2 support
- Connection pooling
- Automatic retry with backoff
- Rate limiting integration
- Proxy support
- User agent rotation

Example:
    >>> config = ScraperConfig(timeout=30)
    >>> async with HTTPXScraper(config) as scraper:
    ...     content = await scraper.scrape("https://example.com")
    ...     print(content.title)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse

from sentimatrix.core.config import ScraperConfig
from sentimatrix.core.exceptions import (
    ScraperError,
    ScraperTimeoutError,
    ScraperConnectionError,
    HTTPError,
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
    RetryHandler,
    DeviceType,
    extract_domain,
)


# Lazy import for httpx
_httpx = None


def _get_httpx():
    """Lazy import httpx."""
    global _httpx
    if _httpx is None:
        try:
            import httpx
            _httpx = httpx
        except ImportError as e:
            raise ImportError(
                "httpx is required for HTTPXScraper. "
                "Install with: pip install httpx"
            ) from e
    return _httpx


# Lazy import for BeautifulSoup
_bs4 = None


def _get_bs4():
    """Lazy import BeautifulSoup."""
    global _bs4
    if _bs4 is None:
        try:
            from bs4 import BeautifulSoup
            _bs4 = BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "beautifulsoup4 is required for HTML parsing. "
                "Install with: pip install beautifulsoup4 lxml"
            ) from e
    return _bs4


class HTTPXScraper(BaseScraperProvider):
    """
    Async HTTP scraper using HTTPX.

    Provides high-performance web scraping for static HTML content
    with built-in support for:
    - Connection pooling and HTTP/2
    - Rate limiting per domain
    - Proxy rotation
    - User agent rotation
    - Automatic retry with exponential backoff
    - Cookie persistence

    Note: This scraper cannot render JavaScript. For JS-heavy sites,
    use PlaywrightScraper instead.
    """

    def __init__(
        self,
        config: Optional[ScraperConfig] = None,
        rate_limiter: Optional[RateLimiter] = None,
        proxy_manager: Optional[ProxyManager] = None,
        user_agent_rotator: Optional[UserAgentRotator] = None,
    ) -> None:
        """
        Initialize HTTPX scraper.

        Args:
            config: Scraper configuration
            rate_limiter: Optional rate limiter (created if not provided)
            proxy_manager: Optional proxy manager
            user_agent_rotator: Optional user agent rotator
        """
        super().__init__(config)
        self._config: ScraperConfig = config or ScraperConfig()

        # HTTP client
        self._client: Optional[Any] = None

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

        # Retry handler
        self._retry_handler = RetryHandler(
            config=self._config.retry,
            retry_on_status=[429, 500, 502, 503, 504],
        )

        # Session cookies
        self._cookies: Dict[str, str] = {}

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="httpx",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Async HTTP scraper using HTTPX for static HTML content",
            capabilities=ProviderCapabilities(
                javascript_rendering=False,
                screenshots=False,
                pdf_generation=False,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    async def initialize(self) -> None:
        """Initialize the HTTPX client."""
        if self._initialized:
            return

        httpx = _get_httpx()

        # Build client configuration
        client_kwargs = {
            "timeout": httpx.Timeout(
                connect=10.0,
                read=float(self._config.timeout),
                write=10.0,
                pool=10.0,
            ),
            "follow_redirects": True,
            "http2": True,
            "limits": httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        }

        # Add proxy if configured
        if self._proxy_manager:
            proxy_config = self._proxy_manager.get_proxy_for_httpx()
            if proxy_config:
                client_kwargs["proxies"] = proxy_config

        self._client = httpx.AsyncClient(**client_kwargs)
        self._initialized = True

    async def close(self) -> None:
        """Close the HTTPX client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    def _get_headers(self, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Get request headers with user agent."""
        headers = {
            "User-Agent": self._config.user_agent or self._ua_rotator.get_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }

        if custom_headers:
            headers.update(custom_headers)

        return headers

    async def scrape(
        self,
        url: str,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        method: str = "GET",
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScrapedContent:
        """
        Scrape content from a URL.

        Args:
            url: URL to scrape
            wait_for: Ignored (no JS rendering)
            timeout: Request timeout override
            headers: Custom headers
            cookies: Custom cookies
            method: HTTP method
            data: Form data for POST
            json: JSON data for POST
            **kwargs: Additional arguments

        Returns:
            ScrapedContent with page content

        Raises:
            ScraperError: If scraping fails
            ScraperTimeoutError: If request times out
            RateLimitError: If rate limited
        """
        self._ensure_initialized()

        httpx = _get_httpx()
        BeautifulSoup = _get_bs4()

        domain = extract_domain(url)
        start_time = time.perf_counter()

        # Rate limiting
        await self._rate_limiter.acquire(domain)

        # Merge cookies
        request_cookies = {**self._cookies}
        if cookies:
            request_cookies.update(cookies)

        # Get headers with user agent
        request_headers = self._get_headers(headers)

        # Prepare request kwargs
        request_kwargs: Dict[str, Any] = {
            "headers": request_headers,
            "cookies": request_cookies,
        }

        if timeout:
            request_kwargs["timeout"] = float(timeout)

        if data:
            request_kwargs["data"] = data
        elif json:
            request_kwargs["json"] = json

        # Get proxy for this request
        proxy_url = None
        if self._proxy_manager:
            proxy = self._proxy_manager.get_proxy()
            if proxy:
                proxy_url = proxy.url

        try:
            # Execute request with retry
            response = await self._retry_handler.execute(
                self._make_request,
                method,
                url,
                **request_kwargs,
            )

            response_time = (time.perf_counter() - start_time) * 1000

            # Report proxy success
            if proxy_url and self._proxy_manager:
                self._proxy_manager.report_success(proxy_url, response_time)

            # Parse HTML
            html = response.text
            soup = BeautifulSoup(html, "lxml")

            # Extract title
            title = None
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)

            # Extract text content
            # Remove script and style elements
            for element in soup(["script", "style", "noscript", "iframe"]):
                element.decompose()

            text_content = soup.get_text(separator="\n", strip=True)

            # Store response cookies
            for cookie_name, cookie_value in response.cookies.items():
                self._cookies[cookie_name] = cookie_value

            return ScrapedContent(
                url=str(response.url),
                title=title,
                content=text_content,
                html=html,
                status_code=response.status_code,
                response_time_ms=response_time,
                headers=dict(response.headers),
                cookies=dict(response.cookies),
                provider="httpx",
                proxy_used=proxy_url,
                user_agent=request_headers.get("User-Agent"),
            )

        except httpx.TimeoutException as e:
            if proxy_url and self._proxy_manager:
                self._proxy_manager.report_failure(proxy_url)
            raise ScraperTimeoutError(
                url=url,
                timeout=timeout or self._config.timeout,
            ) from e

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                self._rate_limiter.report_429(domain)
                raise RateLimitError(
                    provider="httpx",
                    retry_after=60,
                    message=f"Rate limited by {domain}",
                ) from e
            raise HTTPError(
                url=url,
                status_code=e.response.status_code,
                message=str(e),
            ) from e

        except httpx.ConnectError as e:
            if proxy_url and self._proxy_manager:
                self._proxy_manager.report_failure(proxy_url)
            raise ScraperConnectionError(
                url=url,
                reason=str(e),
            ) from e

        except Exception as e:
            if proxy_url and self._proxy_manager:
                self._proxy_manager.report_failure(proxy_url)
            raise ScraperError(
                provider="httpx",
                message=f"Failed to scrape {url}: {e}",
            ) from e

    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Any:
        """Make HTTP request (for retry handler)."""
        httpx = _get_httpx()

        response = await self._client.request(method, url, **kwargs)

        # Raise for 4xx/5xx status codes
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {response.status_code}",
                request=response.request,
                response=response,
            )

        return response

    async def scrape_batch(
        self,
        urls: List[str],
        concurrency: int = 5,
        **kwargs: Any,
    ) -> List[ScrapedContent]:
        """
        Scrape multiple URLs concurrently.

        Args:
            urls: List of URLs to scrape
            concurrency: Maximum concurrent requests
            **kwargs: Arguments passed to scrape()

        Returns:
            List of ScrapedContent objects
        """
        self._ensure_initialized()

        semaphore = asyncio.Semaphore(concurrency)

        async def scrape_with_semaphore(url: str) -> ScrapedContent:
            async with semaphore:
                try:
                    return await self.scrape(url, **kwargs)
                except Exception as e:
                    # Return error content
                    return ScrapedContent(
                        url=url,
                        content=f"Error: {e}",
                        status_code=0,
                        provider="httpx",
                    )

        tasks = [scrape_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def scrape_reviews(
        self,
        url: str,
        limit: int = 100,
        sort_by: Optional[str] = None,
        selectors: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> List[Review]:
        """
        Extract reviews from a URL.

        This is a generic implementation that requires CSS selectors
        to be provided. For platform-specific review extraction,
        use dedicated platform scrapers.

        Args:
            url: URL to scrape reviews from
            limit: Maximum number of reviews
            sort_by: Ignored (no JS interaction)
            selectors: CSS selectors for review elements:
                - container: Selector for review container
                - text: Selector for review text
                - rating: Selector for rating (optional)
                - author: Selector for author (optional)
                - date: Selector for date (optional)
            **kwargs: Additional arguments

        Returns:
            List of Review objects
        """
        self._ensure_initialized()

        BeautifulSoup = _get_bs4()

        if not selectors:
            raise ValueError(
                "selectors dict required with at least 'container' and 'text' keys"
            )

        content = await self.scrape(url, **kwargs)

        if not content.html:
            return []

        soup = BeautifulSoup(content.html, "lxml")
        reviews: List[Review] = []

        # Find all review containers
        container_selector = selectors.get("container", "div.review")
        containers = soup.select(container_selector)[:limit]

        for i, container in enumerate(containers):
            # Extract text
            text_selector = selectors.get("text", "p")
            text_elem = container.select_one(text_selector)
            text = text_elem.get_text(strip=True) if text_elem else ""

            if not text:
                continue

            # Extract rating
            rating = None
            if "rating" in selectors:
                rating_elem = container.select_one(selectors["rating"])
                if rating_elem:
                    try:
                        # Try to extract numeric rating
                        rating_text = rating_elem.get_text(strip=True)
                        # Common patterns: "4/5", "4 out of 5", "4 stars", etc.
                        import re
                        match = re.search(r"(\d+(?:\.\d+)?)", rating_text)
                        if match:
                            rating = float(match.group(1))
                    except (ValueError, AttributeError):
                        pass

            # Extract author
            author = None
            if "author" in selectors:
                author_elem = container.select_one(selectors["author"])
                if author_elem:
                    author = author_elem.get_text(strip=True)

            # Extract date (parsing deferred to caller)
            timestamp = None

            reviews.append(Review(
                id=f"{extract_domain(url)}-{i}",
                text=text,
                source=url,
                platform=extract_domain(url),
                author=author,
                rating=rating,
                timestamp=timestamp,
            ))

        return reviews

    async def get(self, url: str, **kwargs: Any) -> ScrapedContent:
        """Convenience method for GET requests."""
        return await self.scrape(url, method="GET", **kwargs)

    async def post(
        self,
        url: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ScrapedContent:
        """Convenience method for POST requests."""
        return await self.scrape(url, method="POST", data=data, json=json, **kwargs)

    async def head(self, url: str, **kwargs: Any) -> Dict[str, str]:
        """
        Make HEAD request and return headers.

        Args:
            url: URL to check
            **kwargs: Additional arguments

        Returns:
            Response headers
        """
        self._ensure_initialized()

        domain = extract_domain(url)
        await self._rate_limiter.acquire(domain)

        headers = self._get_headers(kwargs.get("headers"))

        response = await self._client.head(url, headers=headers)
        return dict(response.headers)

    async def download(
        self,
        url: str,
        path: str,
        chunk_size: int = 8192,
        **kwargs: Any,
    ) -> str:
        """
        Download a file from URL.

        Args:
            url: URL to download
            path: Local file path
            chunk_size: Download chunk size
            **kwargs: Additional arguments

        Returns:
            Path to downloaded file
        """
        self._ensure_initialized()

        domain = extract_domain(url)
        await self._rate_limiter.acquire(domain)

        headers = self._get_headers(kwargs.get("headers"))

        async with self._client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()

            with open(path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size):
                    f.write(chunk)

        return path

    def set_cookies(self, cookies: Dict[str, str]) -> None:
        """Set session cookies."""
        self._cookies.update(cookies)

    def clear_cookies(self) -> None:
        """Clear session cookies."""
        self._cookies.clear()

    @property
    def rate_limiter(self) -> RateLimiter:
        """Get the rate limiter."""
        return self._rate_limiter

    @property
    def proxy_manager(self) -> Optional[ProxyManager]:
        """Get the proxy manager."""
        return self._proxy_manager


# Register provider
register_provider("httpx", ProviderType.SCRAPER, HTTPXScraper)
