"""
ScraperAPI Integration

ScraperAPI handles proxies, browsers, and CAPTCHAs for web scraping.

Features:
- 40M+ residential and datacenter IPs
- JavaScript rendering
- CAPTCHA handling
- Geo-targeting (50+ countries)
- Structured data parsing (autoparse)
- Screenshot capture

Pricing: From $49/mo for 100k API credits

API Documentation: https://docs.scraperapi.com/

Example:
    >>> from sentimatrix.providers.scrapers.commercial import ScraperAPIClient
    >>>
    >>> async with ScraperAPIClient(api_key="your_key") as client:
    ...     # Basic scraping
    ...     content = await client.scrape("https://example.com")
    ...
    ...     # With JS rendering
    ...     content = await client.scrape("https://spa.com", render_js=True)
    ...
    ...     # With geo-targeting
    ...     content = await client.scrape("https://site.com", country_code="us")
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from sentimatrix.providers.base import (
    ProviderCapabilities,
    ProviderInfo,
    ProviderType,
)
from sentimatrix.providers.scrapers.commercial.base import (
    BaseCommercialClient,
    CommercialAPIConfig,
    DeviceEmulation,
    OutputFormat,
    ScrapeResult,
    _get_httpx,
)


@dataclass
class ScraperAPIConfig(CommercialAPIConfig):
    """ScraperAPI-specific configuration."""

    # API settings
    api_key: Optional[str] = None

    # Rendering options
    render_js: bool = False
    premium_proxy: bool = False
    ultra_premium: bool = False

    # Targeting
    country_code: Optional[str] = None
    session_number: Optional[int] = None

    # Output options
    autoparse: bool = False
    binary_target: bool = False

    # Request options
    keep_headers: bool = False

    # Async/Webhook options
    webhook_url: Optional[str] = None

    # Credit costs (approximate)
    # - Standard: 1 credit
    # - Render JS: 5 credits
    # - Premium proxy: 10 credits
    # - Ultra premium: 25 credits


class ScraperAPIClient(BaseCommercialClient):
    """
    ScraperAPI client for web scraping.

    ScraperAPI is a simple web scraping API that handles:
    - Proxy rotation (40M+ IPs)
    - Browser fingerprinting
    - CAPTCHA solving
    - JavaScript rendering

    Supports Python, Node.js, PHP, Ruby, Java via SDK.
    """

    SERVICE_NAME = "scraperapi"
    BASE_URL = "https://api.scraperapi.com"

    def __init__(self, config: Optional[ScraperAPIConfig] = None, api_key: Optional[str] = None) -> None:
        """
        Initialize ScraperAPI client.

        Args:
            config: ScraperAPI configuration
            api_key: API key (alternative to config.api_key)
        """
        if config is None:
            config = ScraperAPIConfig()

        if api_key:
            config.api_key = api_key

        super().__init__(config)
        self._scraper_config: ScraperAPIConfig = config

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="scraperapi",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="ScraperAPI - Web scraping with proxy rotation and CAPTCHA handling",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                screenshots=True,
                pdf_generation=False,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    async def _make_request(
        self,
        url: str,
        render_js: bool = False,
        premium_proxy: bool = False,
        ultra_premium: bool = False,
        country_code: Optional[str] = None,
        session_number: Optional[int] = None,
        keep_headers: bool = False,
        autoparse: bool = False,
        binary_target: bool = False,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Make request to ScraperAPI.

        Args:
            url: Target URL to scrape
            render_js: Enable JavaScript rendering (+5 credits)
            premium_proxy: Use premium residential proxies (+10 credits)
            ultra_premium: Use ultra premium proxies (+25 credits)
            country_code: Target country (us, uk, de, etc.)
            session_number: Keep same IP for session
            keep_headers: Preserve custom headers
            autoparse: Return structured JSON data
            binary_target: Return binary content (images, PDFs)
            headers: Custom request headers
            cookies: Custom cookies
            wait_for: CSS selector to wait for (with JS rendering)
            timeout: Request timeout

        Returns:
            ScrapeResult with response data
        """
        httpx = _get_httpx()

        if not self._scraper_config.api_key:
            raise ValueError("ScraperAPI key is required. Set via config.api_key or api_key parameter.")

        # Build API parameters
        params: Dict[str, Any] = {
            "api_key": self._scraper_config.api_key,
            "url": url,
        }

        # Rendering options
        if render_js or self._scraper_config.render_js:
            params["render"] = "true"

        if premium_proxy or self._scraper_config.premium_proxy:
            params["premium"] = "true"

        if ultra_premium or self._scraper_config.ultra_premium:
            params["ultra_premium"] = "true"

        # Targeting options
        target_country = country_code or self._scraper_config.country_code
        if target_country:
            params["country_code"] = target_country

        target_session = session_number or self._scraper_config.session_number
        if target_session is not None:
            params["session_number"] = target_session

        # Output options
        if autoparse or self._scraper_config.autoparse:
            params["autoparse"] = "true"

        if binary_target or self._scraper_config.binary_target:
            params["binary_target"] = "true"

        # Request options
        if keep_headers or self._scraper_config.keep_headers:
            params["keep_headers"] = "true"

        # Wait for selector (with JS rendering)
        if wait_for:
            params["wait_for_selector"] = wait_for

        # Custom headers
        if headers:
            for key, value in headers.items():
                params[f"header_{key}"] = value

        # Custom cookies
        if cookies:
            cookie_string = "; ".join(f"{k}={v}" for k, v in cookies.items())
            params["cookie"] = cookie_string

        # Device emulation
        if self._scraper_config.device == DeviceEmulation.MOBILE:
            params["device_type"] = "mobile"
        elif self._scraper_config.device == DeviceEmulation.TABLET:
            params["device_type"] = "tablet"

        try:
            # Make request
            response = await self._client.get(
                self.BASE_URL,
                params=params,
                timeout=timeout or self._scraper_config.timeout,
            )

            # Calculate credits used (approximate)
            credits = 1
            if render_js or self._scraper_config.render_js:
                credits = 5
            if premium_proxy or self._scraper_config.premium_proxy:
                credits = 10
            if ultra_premium or self._scraper_config.ultra_premium:
                credits = 25

            # Parse response
            content = response.text
            json_data = None

            if autoparse or self._scraper_config.autoparse:
                try:
                    json_data = response.json()
                    content = str(json_data)
                except Exception:
                    pass

            return ScrapeResult(
                url=url,
                content=content,
                status_code=response.status_code,
                html=content if not (binary_target or self._scraper_config.binary_target) else None,
                json_data=json_data,
                headers=dict(response.headers),
                credits_used=credits,
                cost_usd=credits * 0.00049,  # Approximate cost at $49/100k
                provider=self.SERVICE_NAME,
            )

        except httpx.TimeoutException as e:
            return ScrapeResult(
                url=url,
                content="",
                status_code=408,
                error=f"Timeout: {e}",
                provider=self.SERVICE_NAME,
            )
        except Exception as e:
            return ScrapeResult(
                url=url,
                content="",
                status_code=0,
                error=str(e),
                provider=self.SERVICE_NAME,
            )

    async def scrape_async(
        self,
        url: str,
        webhook_url: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Submit async scrape job with webhook callback.

        Args:
            url: URL to scrape
            webhook_url: URL to receive results
            **kwargs: Additional scrape options

        Returns:
            Job submission response
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        params = {
            "api_key": self._scraper_config.api_key,
            "url": url,
            "callback": webhook_url,
        }

        # Add other options
        if kwargs.get("render_js"):
            params["render"] = "true"

        response = await self._client.get(
            f"{self.BASE_URL}/jobs",
            params=params,
        )

        return response.json()

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get ScraperAPI account information.

        Returns:
            Account info including credits remaining
        """
        self._ensure_initialized()

        response = await self._client.get(
            f"{self.BASE_URL}/account",
            params={"api_key": self._scraper_config.api_key},
        )

        return response.json()

    async def take_screenshot(
        self,
        url: str,
        full_page: bool = False,
        **kwargs: Any,
    ) -> bytes:
        """
        Take screenshot of a webpage.

        Args:
            url: URL to screenshot
            full_page: Capture full page or viewport only
            **kwargs: Additional options

        Returns:
            Screenshot as PNG bytes
        """
        self._ensure_initialized()

        params = {
            "api_key": self._scraper_config.api_key,
            "url": url,
            "render": "true",
            "screenshot": "true",
        }

        if full_page:
            params["screenshot_full_page"] = "true"

        response = await self._client.get(
            self.BASE_URL,
            params=params,
        )

        return response.content

    def get_proxy_url(
        self,
        url: str,
        render_js: bool = False,
        country_code: Optional[str] = None,
    ) -> str:
        """
        Get proxy URL for use with other libraries (Scrapy, etc.).

        Args:
            url: Target URL
            render_js: Enable JS rendering
            country_code: Target country

        Returns:
            Formatted proxy URL
        """
        params = {
            "api_key": self._scraper_config.api_key,
            "url": url,
        }

        if render_js:
            params["render"] = "true"

        if country_code:
            params["country_code"] = country_code

        return f"{self.BASE_URL}?{urlencode(params)}"
