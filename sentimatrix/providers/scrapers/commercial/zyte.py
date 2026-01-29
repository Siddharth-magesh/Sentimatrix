"""
Zyte Integration (formerly Scrapinghub/Scrapy Cloud)

Zyte is an all-in-one web scraping platform with:
- Automatic extraction using AI
- Browser rendering at scale
- Smart proxy rotation
- Scrapy Cloud integration

Features:
- Zyte API: Unified endpoint for scraping
- Automatic Extraction: AI-powered data extraction
- Scrapy Cloud: Host Scrapy spiders
- Smart Proxy Manager: Intelligent proxy rotation

Pricing: From $450/mo for Zyte API

API Documentation: https://docs.zyte.com/zyte-api/

Example:
    >>> from sentimatrix.providers.scrapers.commercial import ZyteClient
    >>>
    >>> async with ZyteClient(api_key="your_key") as client:
    ...     # Basic scraping
    ...     result = await client.scrape("https://example.com")
    ...
    ...     # With browser rendering
    ...     result = await client.scrape("https://spa.com", browser_html=True)
    ...
    ...     # Automatic extraction
    ...     product = await client.extract_product("https://shop.com/product/123")
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
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


class ZyteExtractionType(str, Enum):
    """Zyte automatic extraction types."""
    ARTICLE = "article"
    ARTICLE_LIST = "articleList"
    ARTICLE_NAVIGATION = "articleNavigation"
    PRODUCT = "product"
    PRODUCT_LIST = "productList"
    PRODUCT_NAVIGATION = "productNavigation"
    JOB_POSTING = "jobPosting"
    FORUM_THREAD = "forumThread"
    CUSTOM = "custom"


class ZyteAction(str, Enum):
    """Browser actions for Zyte API."""
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    WAIT = "wait"
    WAIT_FOR_SELECTOR = "waitForSelector"
    WAIT_FOR_TIMEOUT = "waitForTimeout"
    SELECT = "select"
    SCREENSHOT = "screenshot"


@dataclass
class ZyteConfig(CommercialAPIConfig):
    """Zyte API configuration."""

    api_key: Optional[str] = None

    # Output options
    http_response_body: bool = True
    http_response_headers: bool = False
    browser_html: bool = False
    screenshot: bool = False
    screenshot_options: Dict[str, Any] = field(default_factory=dict)

    # Extraction options
    automatic_extraction: bool = False
    extraction_type: Optional[ZyteExtractionType] = None

    # Request options
    geo_location: Optional[str] = None
    javascript: bool = False
    js_snippet: Optional[str] = None

    # Actions (browser automation)
    actions: List[Dict[str, Any]] = field(default_factory=list)

    # Session management
    session_context: Optional[str] = None
    session_context_parameters: Dict[str, Any] = field(default_factory=dict)


class ZyteClient(BaseCommercialClient):
    """
    Zyte API client.

    Zyte provides:
    - Unified scraping endpoint
    - AI-powered automatic extraction
    - Browser rendering at scale
    - Scrapy integration

    Uses POST requests to a single endpoint.
    """

    SERVICE_NAME = "zyte"
    BASE_URL = "https://api.zyte.com/v1/extract"

    def __init__(
        self,
        config: Optional[ZyteConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize Zyte client.

        Args:
            config: Zyte configuration
            api_key: API key
        """
        if config is None:
            config = ZyteConfig()

        if api_key:
            config.api_key = api_key

        super().__init__(config)
        self._zyte_config: ZyteConfig = config

    @property
    def info(self) -> ProviderInfo:
        """Get provider information."""
        return ProviderInfo(
            name="zyte",
            provider_type=ProviderType.SCRAPER,
            version="1.0.0",
            description="Zyte API - All-in-one web scraping with AI extraction",
            capabilities=ProviderCapabilities(
                javascript_rendering=True,
                screenshots=True,
                pdf_generation=False,
                proxy_support=True,
                batch_processing=True,
            ),
        )

    def _get_auth(self) -> tuple:
        """Get basic auth credentials (API key as username)."""
        return (self._zyte_config.api_key, "")

    async def _make_request(
        self,
        url: str,
        render_js: bool = False,
        browser_html: bool = False,
        http_response_body: bool = True,
        geo_location: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
        wait_for: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Make request to Zyte API.

        Args:
            url: Target URL
            render_js: Enable JavaScript (via browser)
            browser_html: Get browser-rendered HTML
            http_response_body: Get HTTP response body
            geo_location: Target geolocation
            headers: Custom headers
            cookies: Custom cookies
            wait_for: CSS selector to wait for
            timeout: Request timeout

        Returns:
            ScrapeResult with response
        """
        httpx = _get_httpx()

        if not self._zyte_config.api_key:
            raise ValueError("Zyte API key required.")

        # Build request body
        request_body: Dict[str, Any] = {
            "url": url,
        }

        # Output options
        use_browser = render_js or browser_html or self._zyte_config.browser_html
        if use_browser:
            request_body["browserHtml"] = True
        elif http_response_body or self._zyte_config.http_response_body:
            request_body["httpResponseBody"] = True

        if self._zyte_config.http_response_headers:
            request_body["httpResponseHeaders"] = True

        # Screenshot
        if self._zyte_config.screenshot:
            request_body["screenshot"] = True
            if self._zyte_config.screenshot_options:
                request_body["screenshotOptions"] = self._zyte_config.screenshot_options

        # Geo-targeting
        target_geo = geo_location or self._zyte_config.geo_location
        if target_geo:
            request_body["geolocation"] = target_geo

        # JavaScript execution
        if self._zyte_config.javascript or self._zyte_config.js_snippet:
            request_body["javascript"] = True
            if self._zyte_config.js_snippet:
                request_body["jsSnippet"] = self._zyte_config.js_snippet

        # Wait for selector
        if wait_for:
            if "actions" not in request_body:
                request_body["actions"] = []
            request_body["actions"].append({
                "action": "waitForSelector",
                "selector": {"type": "css", "value": wait_for},
            })

        # Add configured actions
        if self._zyte_config.actions:
            if "actions" not in request_body:
                request_body["actions"] = []
            request_body["actions"].extend(self._zyte_config.actions)

        # Custom headers
        if headers:
            request_body["customHttpRequestHeaders"] = [
                {"name": k, "value": v} for k, v in headers.items()
            ]

        # Session context
        if self._zyte_config.session_context:
            request_body["sessionContext"] = self._zyte_config.session_context
            if self._zyte_config.session_context_parameters:
                request_body["sessionContextParameters"] = self._zyte_config.session_context_parameters

        try:
            response = await self._client.post(
                self.BASE_URL,
                json=request_body,
                auth=self._get_auth(),
                timeout=timeout or self._zyte_config.timeout,
            )

            response.raise_for_status()
            data = response.json()

            # Extract content
            content = ""
            html = None
            screenshot_data = None

            if "browserHtml" in data:
                html = data["browserHtml"]
                content = html
            elif "httpResponseBody" in data:
                # Base64 encoded
                body = base64.b64decode(data["httpResponseBody"]).decode("utf-8", errors="ignore")
                html = body
                content = body

            if "screenshot" in data:
                screenshot_data = base64.b64decode(data["screenshot"])

            return ScrapeResult(
                url=data.get("url", url),
                content=content,
                status_code=data.get("statusCode", 200),
                html=html,
                headers=self._parse_headers(data.get("httpResponseHeaders", [])),
                screenshot=screenshot_data,
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

    def _parse_headers(self, headers_list: List[Dict[str, str]]) -> Dict[str, str]:
        """Parse headers from Zyte format."""
        return {h.get("name", ""): h.get("value", "") for h in headers_list}

    async def extract_product(
        self,
        url: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract product data using AI.

        Args:
            url: Product page URL
            **kwargs: Additional options

        Returns:
            Extracted product data
        """
        return await self._extract(url, ZyteExtractionType.PRODUCT, **kwargs)

    async def extract_product_list(
        self,
        url: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract product list data.

        Args:
            url: Product listing URL
            **kwargs: Additional options

        Returns:
            Extracted products
        """
        return await self._extract(url, ZyteExtractionType.PRODUCT_LIST, **kwargs)

    async def extract_article(
        self,
        url: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract article content.

        Args:
            url: Article URL
            **kwargs: Additional options

        Returns:
            Extracted article data
        """
        return await self._extract(url, ZyteExtractionType.ARTICLE, **kwargs)

    async def extract_article_list(
        self,
        url: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract article list data.

        Args:
            url: Article listing URL
            **kwargs: Additional options

        Returns:
            Extracted articles
        """
        return await self._extract(url, ZyteExtractionType.ARTICLE_LIST, **kwargs)

    async def extract_job_posting(
        self,
        url: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract job posting data.

        Args:
            url: Job posting URL
            **kwargs: Additional options

        Returns:
            Extracted job data
        """
        return await self._extract(url, ZyteExtractionType.JOB_POSTING, **kwargs)

    async def _extract(
        self,
        url: str,
        extraction_type: ZyteExtractionType,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Extract structured data using Zyte AI.

        Args:
            url: Target URL
            extraction_type: Type of extraction
            timeout: Request timeout
            **kwargs: Additional options

        Returns:
            Extracted data
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        request_body: Dict[str, Any] = {
            "url": url,
            extraction_type.value: True,
        }

        # Add geo-location if configured
        if self._zyte_config.geo_location:
            request_body["geolocation"] = self._zyte_config.geo_location

        response = await self._client.post(
            self.BASE_URL,
            json=request_body,
            auth=self._get_auth(),
            timeout=timeout or self._zyte_config.timeout,
        )

        response.raise_for_status()
        data = response.json()

        # Return extracted data
        return data.get(extraction_type.value, data)

    async def take_screenshot(
        self,
        url: str,
        full_page: bool = True,
        format: str = "png",
        **kwargs: Any,
    ) -> bytes:
        """
        Take screenshot of a webpage.

        Args:
            url: URL to screenshot
            full_page: Capture full page
            format: Image format (png or jpeg)
            **kwargs: Additional options

        Returns:
            Screenshot bytes
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        request_body: Dict[str, Any] = {
            "url": url,
            "screenshot": True,
            "screenshotOptions": {
                "fullPage": full_page,
                "format": format,
            },
        }

        response = await self._client.post(
            self.BASE_URL,
            json=request_body,
            auth=self._get_auth(),
        )

        response.raise_for_status()
        data = response.json()

        if "screenshot" in data:
            return base64.b64decode(data["screenshot"])

        raise ValueError("No screenshot in response")

    async def execute_actions(
        self,
        url: str,
        actions: List[Dict[str, Any]],
        **kwargs: Any,
    ) -> ScrapeResult:
        """
        Execute browser actions on a page.

        Args:
            url: Target URL
            actions: List of actions to execute
            **kwargs: Additional options

        Returns:
            ScrapeResult after actions
        """
        self._ensure_initialized()
        httpx = _get_httpx()

        request_body: Dict[str, Any] = {
            "url": url,
            "browserHtml": True,
            "actions": actions,
        }

        response = await self._client.post(
            self.BASE_URL,
            json=request_body,
            auth=self._get_auth(),
        )

        response.raise_for_status()
        data = response.json()

        return ScrapeResult(
            url=data.get("url", url),
            content=data.get("browserHtml", ""),
            status_code=data.get("statusCode", 200),
            html=data.get("browserHtml"),
            provider=self.SERVICE_NAME,
        )

    def build_action(
        self,
        action_type: ZyteAction,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Build an action dict for browser automation.

        Args:
            action_type: Type of action
            **params: Action parameters

        Returns:
            Action dictionary
        """
        action = {"action": action_type.value}
        action.update(params)
        return action
