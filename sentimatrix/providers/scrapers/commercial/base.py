"""
Base classes for Commercial Scraping API clients.

Provides common functionality for all commercial scraping services:
- Async HTTP client management
- Rate limiting
- Error handling
- Response parsing
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from sentimatrix.providers.base import (
    BaseScraperProvider,
    ProviderCapabilities,
    ProviderInfo,
    ProviderType,
    ScrapedContent,
)


class OutputFormat(str, Enum):
    """Output format for scraped content."""
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    TEXT = "text"


class DeviceEmulation(str, Enum):
    """Device emulation options."""
    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"


@dataclass
class CommercialAPIConfig:
    """Base configuration for commercial scraping APIs."""

    api_key: Optional[str] = None
    timeout: int = 60
    retries: int = 3
    render_js: bool = False
    premium_proxy: bool = False
    country_code: Optional[str] = None
    device: DeviceEmulation = DeviceEmulation.DESKTOP
    output_format: OutputFormat = OutputFormat.HTML

    # Rate limiting
    requests_per_second: float = 5.0

    # Additional options
    keep_headers: bool = False
    session_number: Optional[int] = None
    custom_headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScrapeResult:
    """Result from commercial scraping API."""

    url: str
    content: str
    status_code: int
    html: Optional[str] = None
    json_data: Optional[Dict[str, Any]] = None
    markdown: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    response_time_ms: float = 0.0
    credits_used: int = 1
    cost_usd: float = 0.0
    provider: str = ""

    # Metadata
    screenshot: Optional[bytes] = None
    pdf: Optional[bytes] = None
    error: Optional[str] = None


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
                "httpx is required for commercial scraping APIs. "
                "Install with: pip install httpx"
            ) from e
    return _httpx


class BaseCommercialClient(BaseScraperProvider, ABC):
    """
    Base class for commercial scraping API clients.

    Provides common functionality:
    - Async HTTP client management
    - Rate limiting per API
    - Error handling and retries
    - Response normalization
    """

    SERVICE_NAME: str = "base"
    BASE_URL: str = ""

    def __init__(self, config: Optional[CommercialAPIConfig] = None) -> None:
        """
        Initialize commercial API client.

        Args:
            config: API configuration
        """
        super().__init__()
        self._config = config or CommercialAPIConfig()
        self._client: Optional[Any] = None
        self._last_request_time: float = 0.0
        self._request_interval: float = 1.0 / self._config.requests_per_second

        # Statistics
        self._total_requests: int = 0
        self._total_credits: int = 0
        self._total_cost: float = 0.0

    @property
    @abstractmethod
    def info(self) -> ProviderInfo:
        """Get provider information."""
        pass

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        if self._initialized:
            return

        httpx = _get_httpx()

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self._config.timeout),
            follow_redirects=True,
            http2=True,
        )
        self._initialized = True

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False

    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        now = time.perf_counter()
        elapsed = now - self._last_request_time

        if elapsed < self._request_interval:
            await asyncio.sleep(self._request_interval - elapsed)

        self._last_request_time = time.perf_counter()

    @abstractmethod
    async def _make_request(
        self,
        url: str,
        **kwargs: Any,
    ) -> ScrapeResult:
        """Make API request to scraping service."""
        pass

    async def scrape(
        self,
        url: str,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        render_js: Optional[bool] = None,
        **kwargs: Any,
    ) -> ScrapedContent:
        """
        Scrape content from URL using commercial API.

        Args:
            url: URL to scrape
            wait_for: CSS selector to wait for (if JS rendering enabled)
            timeout: Request timeout override
            headers: Custom headers
            cookies: Custom cookies
            render_js: Enable JavaScript rendering
            **kwargs: Additional API-specific options

        Returns:
            ScrapedContent with scraped data
        """
        self._ensure_initialized()

        # Apply rate limiting
        await self._rate_limit()

        # Build options
        options = {
            "render_js": render_js if render_js is not None else self._config.render_js,
            "wait_for": wait_for,
            "timeout": timeout or self._config.timeout,
            "headers": {**self._config.custom_headers, **(headers or {})},
            "cookies": {**self._config.cookies, **(cookies or {})},
        }
        options.update(kwargs)

        # Make request
        start_time = time.perf_counter()
        result = await self._make_request(url, **options)
        result.response_time_ms = (time.perf_counter() - start_time) * 1000

        # Update statistics
        self._total_requests += 1
        self._total_credits += result.credits_used
        self._total_cost += result.cost_usd

        # Convert to ScrapedContent
        return ScrapedContent(
            url=result.url,
            title=self._extract_title(result.html or result.content),
            content=result.content,
            html=result.html,
            status_code=result.status_code,
            response_time_ms=result.response_time_ms,
            headers=result.headers,
            cookies=result.cookies,
            provider=self.SERVICE_NAME,
        )

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
            **kwargs: Additional options

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
                    return ScrapedContent(
                        url=url,
                        content=f"Error: {e}",
                        status_code=0,
                        provider=self.SERVICE_NAME,
                    )

        tasks = [scrape_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks)

    def _extract_title(self, html: Optional[str]) -> Optional[str]:
        """Extract title from HTML."""
        if not html:
            return None

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "lxml")
            title_tag = soup.find("title")
            return title_tag.get_text(strip=True) if title_tag else None
        except ImportError:
            # Fallback regex extraction
            import re
            match = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
            return match.group(1).strip() if match else None

    @property
    def stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_requests": self._total_requests,
            "total_credits": self._total_credits,
            "total_cost_usd": round(self._total_cost, 4),
            "avg_credits_per_request": (
                round(self._total_credits / self._total_requests, 2)
                if self._total_requests > 0 else 0
            ),
        }

    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self._total_requests = 0
        self._total_credits = 0
        self._total_cost = 0.0

    async def scrape_reviews(
        self,
        url: str,
        limit: int = 100,
        sort_by: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Any]:
        """
        Extract reviews from a URL.

        Commercial APIs typically scrape the HTML and then parse reviews.
        For structured review extraction, use platform-specific scrapers
        or the AI extraction features of services like Zyte or ScrapingBee.

        Args:
            url: URL to scrape reviews from
            limit: Maximum number of reviews
            sort_by: Sort order (ignored for generic scraping)
            **kwargs: Additional parameters

        Returns:
            List of Review-like objects (dicts for commercial APIs)
        """
        # Generic implementation - scrape and return empty list
        # Subclasses can override for platform-specific extraction
        content = await self.scrape(url, **kwargs)

        # Commercial APIs don't have built-in review extraction
        # Use platform-specific scrapers for structured review data
        return []
