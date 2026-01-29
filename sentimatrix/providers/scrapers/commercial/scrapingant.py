"""
ScrapingAnt Integration

ScrapingAnt is a budget-friendly web scraping API with:
- Headless Chrome rendering
- Proxy rotation
- CAPTCHA solving
- Markdown output support

Features:
- JavaScript rendering
- Custom cookies and headers
- Geo-targeting
- Markdown conversion
- Async POST requests

Pricing: From $19/mo for 10k API credits (most affordable option)

API Documentation: https://docs.scrapingant.com/

Example:
    >>> from sentimatrix.providers.scrapers.commercial import ScrapingAntClient
    >>>
    >>> async with ScrapingAntClient(api_key="your_key") as client:
    ...     # Basic scraping
    ...     result = await client.scrape("https://example.com")
    ...
    ...     # Get as markdown
    ...     markdown = await client.scrape_markdown("https://example.com")
    ...
    ...     # With custom JS
    ...     result = await client.scrape(
    ...         "https://spa.com",
    ...         js_snippet="window.scrollTo(0, document.body.scrollHeight)"
    ...     )
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sentimatrix.providers.base import (
    ProviderCapabilities,
    ProviderInfo,
    ProviderType,
)
from sentimatrix.providers.scrapers.commercial.base import (
    BaseCommercialClient,
    CommercialAPIConfig,
    ScrapeResult,
    _get_httpx,
)


@dataclass
class Cookie:
    """Cookie for ScrapingAnt requests."""
    name: str
    value: str
    domain: Optional[str] = None
    path: str = "/"

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        d = {"name": self.name, "value": self.value, "path": self.path}
        if self.domain:
            d["domain"] = self.domain
        return d


@dataclass
class ScrapingAntConfig(CommercialAPIConfig):
    """ScrapingAnt-specific configuration."""

    api_key: Optional[str] = None

    # Browser options
    browser: bool = True  # Use headless Chrome
    return_page_source: bool = False  # Return page source instead of text

    # Proxy options
    proxy_type: str = "datacenter"  # datacenter, residential
    proxy_country: Optional[str] = None

    # JavaScript
    js_snippet: Optional[str] = None
    wait_for_selector: Optional[str] = None

    # Output
    return_text: bool = False  # Return extracted text only


class ScrapingAntClient(BaseCommercialClient):
    """
    ScrapingAnt API client.

    ScrapingAnt provides:
    - Affordable web scraping API
    - JavaScript rendering
    - Proxy rotation
    - Markdown output support

    Best for:
    - Budget-conscious projects
    - Simple scraping needs
    - LLM data preparation (markdown)
    """

    SERVICE_NAME = "scrapingant"
    BASE_URL = "https://api.scrapingant.com/v2"

    def __init__(
        self,
        config: Optional[ScrapingAntConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize ScrapingAnt client.

        Args:
            config: ScrapingAnt configuration
            api_key: API key (token)
        """
        if config is None:
            config = ScrapingAntConfig()

        if api_key:
            config.api_key = api_key

        super().__init__(config)
        self._sa_config: ScrapingAntConfig = config

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="scrapingant",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="ScrapingAnt - Budget-friendly web scraping with JS rendering",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                screenshots=False,
                pdf_generation=False,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    def _get_headers(self) -> Dict[str, str]:
        """Get API headers."""
        return {
            "x-api-key": self._sa_config.api_key,
            "Content-Type": "application/json",
        }

    async def _make_request(
        self,
        url: str,
        render_js: bool = True,
        proxy_country: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        js_snippet: Optional[str] = None,
        return_text: bool = False,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Make request to ScrapingAnt API.

        Args:
            url: Target URL
            render_js: Use browser (always True for ScrapingAnt)
            proxy_country: Target country code
            headers: Custom headers
            cookies: Custom cookies
            wait_for: CSS selector to wait for
            timeout: Request timeout
            js_snippet: JavaScript to execute
            return_text: Return extracted text only

        Returns:
            ScrapeResult with response
        """
        httpx = _get_httpx()

        if not self._sa_config.api_key:
            raise ValueError("ScrapingAnt API key required.")

        # Build request body
        request_body: Dict[str, Any] = {
            "url": url,
            "browser": self._sa_config.browser,
        }

        # Proxy options
        target_country = proxy_country or self._sa_config.proxy_country
        if target_country:
            request_body["proxy_country"] = target_country

        request_body["proxy_type"] = self._sa_config.proxy_type

        # Wait for selector
        target_wait = wait_for or self._sa_config.wait_for_selector
        if target_wait:
            request_body["wait_for_selector"] = target_wait

        # JavaScript snippet
        target_js = js_snippet or self._sa_config.js_snippet
        if target_js:
            # Base64 encode automatically
            request_body["js_snippet"] = base64.b64encode(target_js.encode()).decode()

        # Custom headers
        if headers:
            request_body["headers"] = headers

        # Custom cookies
        if cookies:
            request_body["cookies"] = [
                {"name": k, "value": v} for k, v in cookies.items()
            ]

        # Return options
        if return_text or self._sa_config.return_text:
            request_body["return_text"] = True

        if self._sa_config.return_page_source:
            request_body["return_page_source"] = True

        try:
            response = await self._client.post(
                f"{self.BASE_URL}/general",
                json=request_body,
                headers=self._get_headers(),
                timeout=timeout or self._sa_config.timeout,
            )

            response.raise_for_status()
            data = response.json()

            content = data.get("content", "")
            text = data.get("text", "")

            # Get response cookies
            response_cookies = {}
            for cookie in data.get("cookies", []):
                response_cookies[cookie.get("name", "")] = cookie.get("value", "")

            return ScrapeResult(
                url=data.get("url", url),
                content=text if return_text else content,
                status_code=data.get("status_code", 200),
                html=content,
                headers=data.get("headers", {}),
                cookies=response_cookies,
                credits_used=1,
                cost_usd=0.0019,  # $19/10k credits
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

    async def scrape_markdown(
        self,
        url: str,
        **kwargs: Any,
    ) -> str:
        """
        Scrape URL and return content as markdown.

        Useful for LLM data preparation.

        Args:
            url: Target URL
            **kwargs: Additional options

        Returns:
            Markdown content
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        request_body: Dict[str, Any] = {
            "url": url,
            "browser": True,
        }

        # Add proxy settings
        if self._sa_config.proxy_country:
            request_body["proxy_country"] = self._sa_config.proxy_country

        response = await self._client.post(
            f"{self.BASE_URL}/markdown",
            json=request_body,
            headers=self._get_headers(),
        )

        response.raise_for_status()
        data = response.json()

        return data.get("markdown", data.get("content", ""))

    async def scrape_extended(
        self,
        url: str,
        cookies: Optional[List[Cookie]] = None,
        js_snippet: Optional[str] = None,
        wait_for_selector: Optional[str] = None,
        proxy_country: Optional[str] = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Extended scraping with full control.

        Args:
            url: Target URL
            cookies: List of Cookie objects
            js_snippet: JavaScript to execute
            wait_for_selector: CSS selector to wait for
            proxy_country: Target country
            **kwargs: Additional options

        Returns:
            ScrapeResult with full response data
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        request_body: Dict[str, Any] = {
            "url": url,
            "browser": True,
        }

        if cookies:
            request_body["cookies"] = [c.to_dict() for c in cookies]

        if js_snippet:
            request_body["js_snippet"] = base64.b64encode(js_snippet.encode()).decode()

        if wait_for_selector:
            request_body["wait_for_selector"] = wait_for_selector

        if proxy_country:
            request_body["proxy_country"] = proxy_country

        response = await self._client.post(
            f"{self.BASE_URL}/extended",
            json=request_body,
            headers=self._get_headers(),
        )

        response.raise_for_status()
        data = response.json()

        return ScrapeResult(
            url=data.get("url", url),
            content=data.get("content", ""),
            status_code=data.get("status_code", 200),
            html=data.get("content"),
            headers=data.get("headers", {}),
            cookies={c.get("name", ""): c.get("value", "") for c in data.get("cookies", [])},
            provider=self.SERVICE_NAME,
        )

    async def scrape_async(
        self,
        url: str,
        method: str = "GET",
        body: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Make async HTTP request (POST, PUT, DELETE support).

        Args:
            url: Target URL
            method: HTTP method (GET, POST, PUT, DELETE)
            body: Request body for POST/PUT
            headers: Custom headers
            **kwargs: Additional options

        Returns:
            ScrapeResult with response
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        request_body: Dict[str, Any] = {
            "url": url,
            "method": method,
        }

        if body:
            request_body["body"] = body

        if headers:
            request_body["headers"] = headers

        response = await self._client.post(
            f"{self.BASE_URL}/general",
            json=request_body,
            headers=self._get_headers(),
        )

        response.raise_for_status()
        data = response.json()

        return ScrapeResult(
            url=data.get("url", url),
            content=data.get("content", ""),
            status_code=data.get("status_code", 200),
            html=data.get("content"),
            headers=data.get("headers", {}),
            provider=self.SERVICE_NAME,
        )

    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account credit balance.

        Note: ScrapingAnt doesn't have a dedicated endpoint.
        Check dashboard for credit balance.

        Returns:
            Basic account info
        """
        return {
            "note": "Check ScrapingAnt dashboard for credit balance",
            "dashboard_url": "https://app.scrapingant.com/dashboard",
        }
